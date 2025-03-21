from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from math import floor
from time import perf_counter
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pacing_auction.elimination import ElimStrategy, Subsequent
from pacing_auction.data import (
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

logger = logging.getLogger("auction")


@dataclass
class Auction:
    # Required
    n: int
    m: int

    # Generated
    v: NDArray[np.float64] = field(init=False)
    b: NDArray[np.float64] = field(init=False)
    alpha_q: NDArray[np.uint64] = field(init=False)

    # Optional
    q: int = 1000
    elim: ElimStrategy = Subsequent
    epsilon: float = 0.0

    # Flags
    no_budget: bool = False
    shuffle: bool = False
    threaded: bool = True
    cache_utility: bool = True
    collect_stats: bool = True

    # Controlling sampling
    seed: Optional[int] = None
    v_dist: Optional[Distribution] = None
    b_scaling_dist: Optional[Distribution] = None
    alpha_q_dist: Optional[Distribution] = None

    # Stats and Cache
    stats: dict[str, Any] = field(default_factory=dict, init=False)
    _utility_cache: dict[tuple[int, tuple[np.uint64, ...]], float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.reset()
        self.init_stats()

    def reset(self) -> None:
        if self.seed:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.get_state()[1][0]  # type: ignore

        # v[bidder][auction] is the valuation of `bidder` for `auction`
        v_dist = self.v_dist or Uniform(0, 1)
        self.v = np.array(
            [[v_dist.sample() for _ in range(self.m)] for _ in range(self.n)]
        )

        # budget[bidder] is the budget of `bidder`
        total_v = np.sum(self.v, axis=0)
        b_scaling = np.ones(self.n)
        b_scaling_dist = self.b_scaling_dist or Uniform(1 / self.m, 1)
        if not self.no_budget:
            b_scaling = np.array([b_scaling_dist.sample() for _ in range(self.n)])

        self.b = total_v * b_scaling

        # alpha[bidder], multiples of q
        alpha_q_dist = self.alpha_q_dist or Discrete((self.q - 1,))
        self.alpha_q = np.array([alpha_q_dist.sample() for _ in range(self.n)])

    def init_stats(self) -> None:
        self.stats = {
            "v": self.v.tolist(),
            "b": self.b.tolist(),
            "alpha_q": self.alpha_q.tolist(),
            "util": [list[float]() for _ in range(self.n)],
            "alphas": [list[float]() for _ in range(self.n)],
            "time": perf_counter(),
            "utility_cache_hits": 0,
            "utility_cache_misses": 0,
            "auction_count": 0,
            "social_welfare": list[float](),
        }

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
        Generate bids for all bidders using NumPy vectorized operations
        """
        return (self.v * (self.alpha_q[:, np.newaxis] / self.q)).astype(np.float64)

    # def social_welfare(self) -> float:
    #     return 0.0

    def fpa_allocate(self, bids: NDArray[np.float64]) -> FPAAllocation | Violation:
        """
        First Price Auction procedure, used to compute the allocation of the auction
        """
        spending = np.zeros(self.n)
        allocations: list[Allocation] = []

        max_bids = np.max(bids, axis=0)

        for auction in range(self.m):
            winning_bid = max_bids[auction]

            winning_mask = bids[:, auction] == winning_bid
            winning_bidders: list[int] = [int(i) for i in np.where(winning_mask)[0]]

            if not winning_bidders:
                raise ValueError("No winning bidders")

            num_winners = len(winning_bidders)
            share = float(winning_bid) / num_winners

            for winning_bidder in winning_bidders:
                spending[winning_bidder] += share
                if spending[winning_bidder] > self.b[winning_bidder]:
                    return Violation(winning_bidder, auction)

            allocations.append(Allocation(winning_bidders, auction, float(winning_bid)))

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
            valid_bids = bids.copy()

            match self.fpa_allocate(valid_bids):
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)

    def fpa_utility_numpy(
        self, bids: NDArray[np.float64], main_bidder: int
    ) -> float | Violation:
        """
        Helper function to calculate the utility of the main bidder
        Same as fpa_utility but using NumPy vectorized operations
        This is slower than fpa_utility for small n and m
        """
        spending = np.zeros(self.n)
        utility = 0.0

        max_bids = bids.max(axis=0)

        for auction in range(self.m):
            winning_bid = max_bids[auction]

            winning_mask = bids[:, auction] == winning_bid
            winning_bidders = np.where(winning_mask)[0]

            num_winners = len(winning_bidders)

            share = winning_bid / num_winners
            spending[winning_bidders] += share

            for winning_bidder in winning_bidders:
                if spending[winning_bidder] > self.b[winning_bidder]:
                    return Violation(winning_bidder, auction)

            if winning_mask[main_bidder]:
                x_ij = 1.0 / num_winners
                utility += (self.v[main_bidder][auction] - winning_bid) * x_ij

        return utility

    def fpa_utility(
        self, bids: NDArray[np.float64], main_bidder: int
    ) -> float | Violation:
        """
        Helper function, subprocedure of utility
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
        if new_alpha_q is not None:
            alpha_values = list(self.alpha_q)
            alpha_values[bidder] = np.uint64(new_alpha_q)
            cache_key = (bidder, tuple(alpha_values))
        else:
            cache_key = (bidder, tuple(self.alpha_q))

        # Use cache if enabled and key exists
        if self.cache_utility and cache_key in self._utility_cache:
            if self.collect_stats:
                self.stats["utility_cache_hits"] += 1
            return self._utility_cache[cache_key]

        if self.collect_stats:
            self.stats["utility_cache_misses"] += 1

        # mask[bidder][auction], True if not eliminated
        mask = np.ones((self.n, self.m), dtype=np.int8)

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
                    # Cache the result if caching is enabled
                    if self.cache_utility:
                        self._utility_cache[cache_key] = utility
                    return utility

    def potential_alpha_qs(self, bidder: int) -> set[int]:
        """
        Calculate the potential alpha qs for a bidder to best respond
        """
        new_alpha_qs = set[int]()
        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder or self.v[bidder][auction] == 0:
                    continue

                other_bid: float = self.v[other_bidder][auction] * (
                    self.alpha_q[other_bidder] / self.q
                )

                multiple: float = other_bid / self.v[bidder][auction]
                q_multiple: int = floor(multiple * self.q) + 1
                new_alpha_q: int = min(q_multiple, self.q)
                new_alpha_qs.add(new_alpha_q)
        return new_alpha_qs

    def best_response_threaded(self, bidder: int) -> BestResponse:
        """
        Find the best response for a bidder, given the current state using threads
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_util = curr_util
        max_alpha_q = curr_alpha_q

        new_alpha_qs = self.potential_alpha_qs(bidder)
        futures: list[Future[tuple[int, float]]] = []

        for new_alpha_q in new_alpha_qs:
            f = self.executor.submit(
                lambda a, b: (a, self.utility(b, a)), new_alpha_q, bidder
            )
            futures.append(f)

        for f in as_completed(futures):
            new_alpha_q, new_utility = f.result()
            if new_utility > max_util:
                max_util = new_utility
                max_alpha_q = new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def best_response(self, bidder: int) -> BestResponse:
        """
        Find the best response for a bidder, given the current state
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        new_alpha_qs = self.potential_alpha_qs(bidder)

        for new_alpha_q in new_alpha_qs:
            new_utility = self.utility(bidder, new_alpha_q)
            if new_utility > max_util:
                max_util = new_utility
                max_alpha_q = new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def responses(self) -> AuctionResult:
        """
        Find the Pure Nash Equilibrium or Cycle of the auction
        """
        import sys

        if self.threaded:
            if hasattr(sys, "_is_gil_enabled"):
                if sys._is_gil_enabled():
                    logger.warning("Threaded execution is enabled but GIL is enabled")
            else:
                logger.warning(
                    "Threaded execution is enabled but no GIL is not supported"
                )

        self.executor = ThreadPoolExecutor()

        # Set up stats to be collected
        if self.collect_stats:
            self.init_stats()

        iteration = 1
        seen = {tuple(self.alpha_q): 0}
        order = list(range(self.n))

        together = True

        while True:
            utility_change = False

            # Shuffle the order of bidders if shuffle is enabled
            if self.shuffle and len(order) > 1:
                last_bidder = order[-1]
                np.random.shuffle(order[:-1])
                if order[0] == last_bidder:
                    swap_idx = np.random.randint(1, len(order))
                    order[0], order[swap_idx] = order[swap_idx], order[0]

            responses = self.alpha_q[:]

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

                    # Update the alpha value
                    if together:
                        responses[bidder] = res.new_alpha_q
                    else:
                        self.alpha_q[bidder] = res.new_alpha_q
                else:
                    if self.stats:
                        self.stats["util"][bidder].append(res.old_utility)

            if together:
                self.alpha_q = responses

            # PNE found
            if not utility_change:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                self.executor.shutdown(wait=True)
                result = PNE(iteration, self.allocate(), stats=deepcopy(self.stats))
                # self.stats = dict()
                return result
            # Cycle detection
            t = tuple(self.alpha_q)
            if t in seen:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                    self.stats["cycle_length"] = iteration - seen[t]
                self.executor.shutdown(wait=True)
                result = Cycle(iteration, stats=deepcopy(self.stats))
                # self.stats = dict()
                return result
            seen[t] = iteration

            iteration += 1
