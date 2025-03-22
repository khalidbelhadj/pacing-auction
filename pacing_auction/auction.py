from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import multiprocessing
import sys
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from math import floor
import threading
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
    Gaussian,
    Pareto,
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
    collect_stats: bool = True

    # Controlling sampling
    seed: Optional[int] = None
    v_dist: Optional[Distribution] = None
    b_scaling_dist: Optional[Distribution] = None
    alpha_q_dist: Optional[Distribution] = None

    # Stats
    stats: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.seed:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.get_state()[1][0]  # type: ignore

        # v[bidder][auction] is the valuation of `bidder` for `auction`
        v_dist = self.v_dist or Pareto(self.n)
        self.v = np.array(
            [[v_dist.sample() for _ in range(self.m)] for _ in range(self.n)]
        )

        # budget[bidder] is the budget of `bidder`
        total_v: NDArray[np.float64] = np.sum(self.v, axis=1)

        b_scaling: NDArray[np.float64] = np.full(self.n, np.inf)
        b_scaling_dist = self.b_scaling_dist or Gaussian(1 / self.m, 1)
        if not self.no_budget:
            b_scaling = np.array([b_scaling_dist.sample() for _ in range(self.n)])

        self.b = total_v * b_scaling

        # alpha[bidder], multiples of q
        alpha_q_dist = self.alpha_q_dist or Discrete((self.q,))
        self.alpha_q = np.array([int(alpha_q_dist.sample()) for _ in range(self.n)])

        self.init_stats()

    def init_stats(self) -> None:
        """
        Initialise the stats dictionary
        """
        self.stats = {
            "utility": [list[float]() for _ in range(self.n)],
            "alpha_q": [list[float]() for _ in range(self.n)],
            "time": perf_counter(),
            "auction_count": 0,
            "social_welfare": list[float](),
            "liquid_welfare": list[float](),
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

    def fpa_allocate(
        self, bids: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | Violation:
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

        x: NDArray[np.float64] = np.zeros((self.n, self.m))
        p: NDArray[np.float64] = np.zeros(self.m)

        for auction in range(self.m):
            winning_bidders: list[int] = []
            winning_bid = -1.0

            for bidder in range(self.n):
                bid = bids[bidder][auction]
                if bid > winning_bid:
                    winning_bidders = [bidder]
                    winning_bid = bid
                elif bid == winning_bid:
                    winning_bidders.append(bidder)

            for winning_bidder in winning_bidders:
                x_ij = 1 / len(winning_bidders)
                spending[winning_bidder] += winning_bid * x_ij
                if spending[winning_bidder] > self.b[winning_bidder]:
                    return Violation(winning_bidder, auction)

                x[winning_bidder][auction] = x_ij

            p[auction] = winning_bid

        return x, p

    def allocate(
        self, adjustment: Optional[tuple[int, int]] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the allocation of the auction given the current state

        Parameters:
            adjustment: Optional[tuple[int, float]] = None
                The adjustment to the alpha values for a bidder

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]
            The tuple of prices and allocations for the auction
        """
        mask = np.ones((self.n, self.m), dtype=np.int8)

        # Precompute bids
        bids = self.bids()
        if adjustment is not None:
            bidder, new_alpha_q = adjustment
            bids[bidder] = self.v[bidder] * (new_alpha_q / self.q)

        while True:
            valid_bids = np.multiply(mask, bids)
            match self.fpa_allocate(valid_bids):
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)
                case x, p:
                    return x, p

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

    def value(
        self, x: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculate the value of each bidder given the allocation and prices

        Parameters:
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
            p: NDArray[np.float64]
                The prices of the auctions (m)
        Returns:
            NDArray[np.float64]
            The value vector
        """
        return np.sum(np.multiply(self.v, x), axis=1)

    def utility(
        self, x: NDArray[np.float64], p: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculate the utility of each bidder given the allocation and prices

        Parameters:
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
            p: NDArray[np.float64]
                The prices of the auctions (m)
        Returns:
            NDArray[np.float64]
            The utility vector
        """
        return np.sum(np.multiply(np.subtract(self.v, p), x), axis=1)

    def social_welfare(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """
        Calculate the social welfare of the auction given the allocation and prices

        Parameters:
            p: NDArray[np.float64]
                The prices of the auctions (m)
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
        Returns:
            float
            The social welfare of the auction
        """
        return np.sum(np.multiply(self.v, x))

    def liquid_welfare(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """
        Calculate the liquid welfare of the auction given the allocation and prices

        Parameters:
            p: NDArray[np.float64]
                The prices of the auctions (m)
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
        Returns:
            float
            The liquid welfare of the auction
        """
        return np.sum(np.minimum(self.b, np.sum(np.multiply(self.v, x), axis=1)))

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
        initial_x, initial_p = self.allocate()
        curr_util = self.utility(initial_x, initial_p)[bidder]

        max_util = curr_util
        max_alpha_q = curr_alpha_q

        new_alpha_qs = self.response_candidates(bidder)
        futures: list[Future[tuple[int, float]]] = []

        for new_alpha_q in new_alpha_qs:

            def func(alpha_q: int, bidder: int) -> tuple[int, float]:
                # return alpha_q, self.utility(*self.allocate((bidder, alpha_q)))[bidder]
                x, p = self.allocate((bidder, alpha_q))
                return alpha_q, self.utility(x, p)[bidder]

            future = self.executor.submit(func, new_alpha_q, bidder)
            futures.append(future)

        for future in as_completed(futures):
            new_alpha_q, new_utility = future.result()
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
        curr_util = self.utility(*self.allocate())[bidder]

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        new_alpha_qs = self.response_candidates(bidder)

        for new_alpha_q in new_alpha_qs:
            new_utility = self.utility(*self.allocate((bidder, new_alpha_q)))[bidder]
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
                np.random.shuffle(order)
                if order[0] == last_bidder:
                    swap_idx = np.random.randint(1, len(order))
                    order[0], order[swap_idx] = order[swap_idx], order[0]

            for bidder in order:
                if self.stats:
                    self.stats["alpha_q"][bidder].append(self.alpha_q[bidder])

                res = (
                    self.best_response_threaded(bidder)
                    if self.threaded
                    else self.best_response(bidder)
                )

                # Utility increased by more than epsilon
                if res.new_utility > res.old_utility + self.epsilon:
                    if self.stats:
                        self.stats["utility"][bidder].append(res.new_utility)
                    utility_change = True

                    # Update the alpha value
                    self.alpha_q[bidder] = res.new_alpha_q
                else:
                    if self.stats:
                        self.stats["utility"][bidder].append(res.old_utility)

            if self.stats:
                self.stats["social_welfare"].append(
                    self.social_welfare(*self.allocate())
                )
                self.stats["liquid_welfare"].append(
                    self.liquid_welfare(*self.allocate())
                )

            # PNE found
            if not utility_change:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                self.executor.shutdown(wait=True)
                result = PNE(iteration, *self.allocate(), stats=deepcopy(self.stats))
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


def auctions(count: int, *args, **kwargs):
    for _ in range(count):
        yield Auction(*args, **kwargs)


def responses(auction):
    return auction.responses()


def responses_with_retry(auction):
    return auction.responses_with_retry()


def collect(
    results,
    count: int,
    *args,
    on_complete=None,
    break_cycles=False,
    max_cycles=3,
    alt_responses=3,
    **kwargs,
):
    """
    Collect the results of multiple auction simulations.

    Parameters:
        results: list
            List to store the simulation results (BRDResult objects)
        count: int
            Number of auctions to run
        *args
            Positional arguments to pass to the Auction constructor
        on_complete: Optional[Callable]
            Callback function called after each auction completes
            with signature (current_index, total_count, result)
        break_cycles: bool = False
            If True, attempts to break out of cycles by trying alternative responses
        max_cycles: int = 3
            Maximum number of cycles to try breaking out of before giving up
        alt_responses: int = 3
            Number of alternative responses to try for each bidder
        **kwargs
            Keyword arguments to pass to the Auction constructor

    Returns:
        list
            The same results list that was passed in, now populated
    """
    i = 0
    i_lock = threading.Lock()

    with multiprocessing.Pool() as pool:
        if break_cycles:
            # Create a wrapper function to pass parameters to responses_with_retry
            def responses_with_retry_params(auction):
                return auction.responses_with_retry(
                    max_cycles=max_cycles, alt_responses=alt_responses
                )

            # Use the wrapper function
            for auction in auctions(count, *args, **kwargs):
                result = responses_with_retry_params(auction)
                with i_lock:
                    i += 1
                    if on_complete:
                        on_complete(i, count, result)
                results.append(result)
        else:
            # Use the standard multiprocessing approach for non-retry mode
            for result in pool.imap_unordered(
                responses, auctions(count, *args, **kwargs)
            ):
                with i_lock:
                    i += 1
                    if on_complete:
                        on_complete(i, count, result)
                results.append(result)

    return results
