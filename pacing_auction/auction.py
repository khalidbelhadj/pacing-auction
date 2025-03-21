from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import sys
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
    BRDResult,
    PNE,
    Cycle,
    Allocation,
    Uniform,
    Violation,
    BestResponse,
)

logger = logging.getLogger("auction")


@dataclass
class Auction:
    """
    A first-price pacing auction with budget constraints.

    This class implements a first-price auction mechanism where bidders have budget
    constraints and use pacing multipliers (alpha) to scale down their bids. The auction
    finds pure Nash equilibria (PNE) or cycles in bidder strategies.

    Required Parameters:
        n: int
            Number of bidders
        m: int
            Number of auctions/items

    Optional Parameters:
        q: int = 1000
            The granularity of pacing multipliers (alpha). Higher values allow for finer adjustments.
        elim: ElimStrategy = Subsequent
            Strategy for eliminating bidders who violate budget constraints.
        epsilon: float = 0.0
            Minimum utility improvement threshold for a bidder to change strategy.

    Flags:
        no_budget: bool = False
            If True, ignore budget constraints.
        shuffle: bool = False
            If True, randomize the order in which bidders update their strategies.
        threaded: bool = True
            If True, use multithreading for best response calculations.
        cache_utility: bool = True
            If True, cache utility calculations to improve performance.
        collect_stats: bool = True
            If True, collect statistics during the auction process.

    Sampling Control Parameters:
        seed: Optional[int] = None
            Random seed for reproducibility.
        v_dist: Optional[Distribution] = None
            Distribution for sampling bidder valuations, defaults to Uniform(0, 1).
        b_scaling_dist: Optional[Distribution] = None
            Distribution for sampling budget scaling factors, defaults to Uniform(1/m, 1).
        alpha_q_dist: Optional[Distribution] = None
            Distribution for initial alpha_q values, defaults to Discrete(q-1).
    """

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

        self.init_stats()

    def init_stats(self) -> None:
        """
        Initialise the stats dictionary
        """
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

        Parameters:
            file_path: str
                The path to the JSON file containing the simulation data

        Returns:
            Auction
            The auction instance loaded from the JSON file
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

        Parameters:
            file_path: str
                The path to the JSON file to save the simulation data
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file)

    def bids(self) -> NDArray[np.float64]:
        """
        Generate bids for all bidders

        Returns:
            The bids for all bidders as a NumPy matrix of shape (n, m)
        """
        return (self.v * (self.alpha_q[:, np.newaxis] / self.q)).astype(np.float64)

    def fpa_allocate(self, bids: NDArray[np.float64]) -> list[Allocation] | Violation:
        """
        First Price Auction procedure, used to compute the allocation of the auction
        Returns a list of allocations or a Violation if a bidder is over budget

        Parameters:
            bids: NDArray[np.float64]
                The bids for all bidders as a NumPy matrix of shape (n, m)

        Returns:
            FPAAllocation | Violation
            The allocation of the auction or a violation if a bidder is over budget
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

        return allocations

    def allocate(
        self, adjustment: Optional[tuple[int, float]] = None
    ) -> list[Allocation]:
        """
        Compute the allocation of the auction given the current state

        Parameters:
            adjustment: Optional[tuple[int, float]] = None
                The adjustment to the alpha values for a bidder

        Returns:
            list[Allocation]
            The allocation of the auction
        """
        mask = np.ones((self.n, self.m), dtype=np.int8)

        # Precompute bids
        bids = self.bids()
        if adjustment is not None:
            bidder, new_alpha_q = adjustment
            bids[bidder] = self.v[bidder] * (new_alpha_q / self.q)

        while True:
            valid_bids = bids.copy()

            match self.fpa_allocate(valid_bids):
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)
                case allocations:
                    return allocations

    def fpa_utility(
        self, bids: NDArray[np.float64], main_bidder: int
    ) -> float | Violation:
        """
        Helper function, sub-procedure of `utility`
        Computes the utility of a bidder, but returns a Violation if a bidder is over budget

        Parameters:
            bids: NDArray[np.float64]
                The bids for all bidders as a NumPy matrix of shape (n, m)
                We provide these as an argument as they might be adjusted or precomputed
            main_bidder: int
                The index of the bidder to calculate the utility for
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

        Parameters:
            bidder: int
                The index of the bidder to calculate the utility for
            new_alpha_q: Optional[int] = None
                The new alpha value to use for the bidder

        Returns:
            float
            The utility of the bidder
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

    def response_candidates(self, bidder: int) -> set[int]:
        """
        Calculate the potential alpha qs for a bidder to best respond

        Parameters:
            bidder: int
                The index of the bidder to calculate the potential alpha qs for

        Returns:
            set[int]
            The potential alpha qs for the bidder to best respond
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

        Parameters:
            bidder: int
                The index of the bidder to find the best response for

        Returns:
            BestResponse
            The best response for the bidder
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_util = curr_util
        max_alpha_q = curr_alpha_q

        new_alpha_qs = self.response_candidates(bidder)
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

        Parameters:
            bidder: int
                The index of the bidder to find the best response for

        Returns:
            BestResponse
            The best response for the bidder
        """
        curr_alpha_q = self.alpha_q[bidder]
        curr_util = self.utility(bidder)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        new_alpha_qs = self.response_candidates(bidder)

        for new_alpha_q in new_alpha_qs:
            new_utility = self.utility(bidder, new_alpha_q)
            if new_utility > max_util:
                max_util = new_utility
                max_alpha_q = new_alpha_q

        return BestResponse(bidder, max_alpha_q, max_util, curr_util)

    def responses(self) -> BRDResult:
        """
        Find the Pure Nash Equilibrium or Cycle of the auction, given the current state

        Returns:
            BRDResult
            The result of the best-response dynamics process
        """

        if self.threaded:
            if hasattr(sys, "_is_gil_enabled"):
                if getattr(sys, "_is_gil_enabled")():
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

        while True:
            utility_change = False

            # Shuffle the order of bidders if shuffle is enabled
            if self.shuffle and len(order) > 1:
                last_bidder = order[-1]
                np.random.shuffle(order[:-1])
                if order[0] == last_bidder:
                    swap_idx = np.random.randint(1, len(order))
                    order[0], order[swap_idx] = order[swap_idx], order[0]

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
                    self.alpha_q[bidder] = res.new_alpha_q
                else:
                    if self.stats:
                        self.stats["util"][bidder].append(res.old_utility)

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
