from dataclasses import dataclass, field
import json
import logging
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from math import floor
from time import perf_counter, time, time_ns
from typing import Any, Callable, Generator, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

import honours_project.elimination as elimination
from honours_project.data import (
    Discrete,
    Distribution,
    AuctionResult,
    PNE,
    Cycle,
    Allocation,
    Uniform,
    Violation,
    FPAAllocation,
    BestResponse,
)

logger = logging.getLogger("simulation")


@dataclass
class Auction:
    # Required
    n: int
    m: int

    # Optional
    q: int = 1000
    elim: elimination.ElimStrategy = elimination.Subsequent
    epsilon: float = 0.0

    # Flags
    no_budget: bool = False
    shuffle: bool = False
    threaded: bool = True
    collect_stats: bool = True

    # Controlling sampling
    seed: Optional[int] = None
    v_dist: Optional[Distribution] = None
    alpha_q_dist: Optional[Distribution] = None

    def __post_init__(self) -> None:
        if self.seed:
            np.random.seed(self.seed)
        self.seed = self.seed if self.seed is not None else np.random.get_state()[1][0]  # type: ignore

        # v[bidder][auction] is the valuation of `bidder` for `auction`
        v_dist = self.v_dist or Uniform(0, 1)
        self.v: NDArray[np.float64] = np.array(
            [[v_dist.sample() for _ in range(self.m)] for _ in range(self.n)]
        )

        # budget[bidder] is the budget of `bidder`
        if self.no_budget:
            self.b = np.full(self.n, np.inf)
        else:
            self.b: NDArray[np.float64] = np.array(
                [np.max(self.v[i]) for i in range(self.n)]
            )

        # alpha[bidder], multiples of q
        alpha_q_dist = self.alpha_q_dist or Discrete((self.q - 1,))
        self.alpha_q: NDArray[np.uint64] = np.array(
            [alpha_q_dist.sample() for _ in range(self.n)]
        )

    @classmethod
    def load(cls, file_path: str) -> "Auction":
        """
        Load a simulation from a JSON file
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)

        return instance

    def save(self, file_path: str) -> None:
        """
        Save a simulation to a JSON file
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file)

    def bids(self) -> NDArray[np.float64]:
        """
        Generate bids for all bidders
        """
        return (self.v * (self.alpha_q[:, np.newaxis] / self.q)).astype(np.float64)

    def fpa(self, bids: NDArray[np.float64]) -> FPAAllocation | Violation:
        """
        First Price Auction procedure, used to compute the allocation of the auction
        """
        spending = np.zeros(self.n)
        allocations: list[Allocation] = []

        for auction in range(self.m):
            winning_bidders = []
            winning_bid = -1

            for bidder in range(self.n):
                bid = bids[bidder][auction]
                if bid > winning_bid:
                    winning_bidders = [bidder]
                    winning_bid = bid
                elif bid == winning_bid:
                    winning_bidders.append(bidder)

            assert len(winning_bidders) != 0
            assert winning_bid != -1

            for winning_bidder in winning_bidders:
                spending[winning_bidder] += winning_bid / len(winning_bidders)
                if spending[winning_bidder] > self.b[winning_bidder]:
                    return Violation(winning_bidder, auction)

            allocations.append(Allocation(winning_bidders, auction, winning_bid))

        return FPAAllocation(allocations, -1)

    def allocate(
        self, adjustment: Optional[tuple[int, float]] = None
    ) -> list[Allocation]:
        """
        Compute the allocation of the auction given the current state
        """
        # mask[bidder][auction], True if not eliminated
        mask = np.ones((self.n, self.m), dtype=np.int8)

        # Precompute bids
        bids = self.bids()
        if adjustment is not None:
            bidder, new_alpha_q = adjustment
            bids[bidder] = self.v[bidder] * (new_alpha_q / self.q)

        while True:
            valid_bids = np.multiply(mask, bids)
            match self.fpa(valid_bids):
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)

    def fpa_utility(
        self, bids: NDArray[np.float64], main_bidder: int
    ) -> float | Violation:
        """
        Helper function to calculate the utility
        """
        spending = np.zeros(self.n)
        utility = 0

        for auction in range(self.m):
            winning_bidders = []
            winning_bid = -1

            for bidder in range(self.n):
                bid = bids[bidder][auction]
                if bid > winning_bid:
                    winning_bidders = [bidder]
                    winning_bid = bid
                elif bid == winning_bid:
                    winning_bidders.append(bidder)

            for winning_bidder in winning_bidders:
                spending[winning_bidder] += winning_bid / len(winning_bidders)
                if spending[winning_bidder] > self.b[winning_bidder]:
                    return Violation(winning_bidder, auction)

                if winning_bidder == main_bidder:
                    x_ij = 1 / len(winning_bidders)
                    utility += (self.v[main_bidder][auction] - winning_bid) * x_ij
        return utility

    def utility(self, bidder: int, new_alpha_q: Optional[int] = None) -> float:
        """
        Calculate the utility of a bidder by simulating the auction procedure.
        The alpha values are adjusted to new_alpha_q if provided
        """
        # mask[bidder][auction], True if not eliminated
        mask = np.ones((self.n, self.m), dtype=bool)

        # Precompute bids
        bids = self.bids()
        if new_alpha_q is not None:
            bids[bidder] = self.v[bidder] * (new_alpha_q / self.q)

        while True:
            valid_bids = np.multiply(mask, bids)
            match self.fpa_utility(valid_bids, bidder):
                case Violation(violating_bidder, auction):
                    self.elim.eliminate(violating_bidder, auction, mask)
                case utility:
                    return utility

    def best_response_auction(self, bidder: int, auction: int) -> BestResponse:
        """
        Find the best response for a bidder in a specific auction, given the current state
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        for other_bidder in range(self.n):
            if other_bidder == bidder or self.v[bidder][auction] == 0:
                continue

            other_bid = self.v[other_bidder][auction] * (
                self.alpha_q[other_bidder] / self.q
            )

            multiple = other_bid / self.v[bidder][auction]
            # Add 1 to outbid the other bidder by 1/q
            q_multiple = int(floor(multiple * self.q) + 1)
            new_alpha_q = min(q_multiple, self.q)

            new_util = self.utility(bidder, new_alpha_q)

            if new_util >= max_util:
                max_util = new_util
                max_alpha_q = new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def best_response_threaded(self, bidder: int) -> BestResponse:
        """
        Find the best response for a bidder, given the current state using threads

        Performance improvement over best_response when using python>=3.13 with -Xgil=0 flag
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        futures: list[Future[BestResponse]] = []
        for auction in range(self.m):
            f = self.executor.submit(self.best_response_auction, bidder, auction)
            futures.append(f)

        # Accumulate results from each auction
        max_util = curr_util
        max_alpha_q = curr_alpha_q
        for f in as_completed(futures):
            res = f.result()
            if res.new_utility > max_util:
                max_util = res.new_utility
                max_alpha_q = res.new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def best_response(self, bidder: int) -> BestResponse:
        """
        Find the best response for a biddder, given the current state
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        for auction in range(self.m):
            res = self.best_response_auction(bidder, auction)
            if res.new_utility >= max_util:
                max_util = res.new_utility
                max_alpha_q = res.new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def run(self) -> AuctionResult:
        """
        Run the simulation until a PNE or cycle is found
        """
        self.executor = ThreadPoolExecutor()

        # Set up stats to be collected
        self.stats: Optional[dict[str, Any]] = None
        if self.collect_stats:
            self.stats = dict()
            self.stats["util"] = [list[float]() for _ in range(self.n)]
            self.stats["alphas"] = [list[float]() for _ in range(self.n)]
            self.stats["win_counts"] = [[0] * self.m for _ in range(self.n)]
            self.stats["time"] = perf_counter()

        iteration = 1
        seen = {tuple(self.alpha_q): 0}
        order = list(range(self.n))

        while True:
            utility_change = False

            # HACK: making sure that the last responded bidder doesn't respond first
            last = order[-1]
            while order[0] != last:
                np.random.shuffle(order)

            for bidder in order:
                if self.stats:
                    self.stats["alphas"][bidder].append(self.alpha_q[bidder])

                res = (
                    self.best_response_threaded(bidder)
                    if self.threaded
                    else self.best_response(bidder)
                )

                # Utility increased by more than epsilon
                if res.new_utility > res.old_utility + self.epsilon:
                    if self.stats:
                        self.stats["util"][bidder].append(res.new_utility)
                    utility_change = True
                    self.alpha_q[bidder] = res.new_alpha_q
                else:
                    if self.stats:
                        self.stats["util"][bidder].append(res.old_utility)

            # PNE found
            if not utility_change:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                self.executor.shutdown(wait=True)
                return PNE(iteration, self.allocate(), stats=self.stats or dict())

            # Cycle detection
            t = tuple(self.alpha_q)
            if t in seen:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                    self.stats["cycle_length"] = iteration - seen[t]
                self.executor.shutdown(wait=True)
                return Cycle(iteration, stats=self.stats or dict())
            seen[t] = iteration

            iteration += 1
