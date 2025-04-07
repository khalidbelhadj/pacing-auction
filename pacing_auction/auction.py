from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import sys
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from math import floor
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray, ArrayLike

from pacing_auction.elimination import ElimStrategy, Subsequent
from pacing_auction.generator import AuctionGenerator, CompleteAuctionGenerator
from pacing_auction.data import (
    BRDResult,
    PNE,
    Cycle,
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
        seed: int | None = None
            Random seed for reproducibility.
    """

    # Required
    n: int
    m: int

    # Generated
    _v: NDArray[np.float64] = field(init=False, repr=False)
    _b: NDArray[np.float64] = field(init=False, repr=False)
    _alpha_q: NDArray[np.uint64] = field(init=False, repr=False)

    # Optional
    q: int = 1000
    elim: ElimStrategy = Subsequent
    epsilon: float = 0.0

    # Flags
    no_budget: bool = False
    shuffle: bool = False
    threaded: bool = True
    collect_stats: bool = True

    # Misc
    stats: dict[str, Any] = field(default_factory=dict, init=False)
    generator: AuctionGenerator = CompleteAuctionGenerator()
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("Number of bidders must be positive")
        if self.m <= 0:
            raise ValueError("Number of items must be positive")
        if self.q <= 0:
            raise ValueError("Granularity of pacing multipliers must be positive")
        if self.epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        self.v, self.b = self.generator.generate(self.n, self.m, self.rng)

        if self.no_budget:
            self._b = np.full(self.n, float("inf"))

        self._alpha_q = np.array(
            [np.round(self.rng.uniform(1, self.q)) for _ in range(self.n)]
        )

        self.init_stats()

    def init_stats(self) -> None:
        """
        Initialise the stats dictionary
        """
        self.stats = {
            "time": perf_counter(),
            "cycle_length": 0,
            "utility": [list[float]() for _ in range(self.n)],
            "alpha_q": [list[float]() for _ in range(self.n)],
            "social_welfare": list[float](),
            "liquid_welfare": list[float](),
            "revenue": list[float](),
        }

    @property
    def v(self):
        """
        Value matrix representing the bidders' valuations for items.

        Returns:
            NDArray[np.float64]: Matrix of shape (n, m) where v[i, j] is
            bidder i's value for item j.
        """
        return self._v

    @v.setter
    def v(self, value: ArrayLike):
        try:
            array = np.array(value, dtype=np.float64)
            if array.shape != (self.n, self.m):
                raise ValueError(
                    f"v must have shape ({self.n}, {self.m}), got {array.shape}"
                )
            self._v = array
        except (ValueError, TypeError):
            raise TypeError(
                f"v must be convertible to a numpy array with shape ({self.n}, {self.m})"
            )

    @property
    def b(self):
        """
        Budget vector representing the bidders' spending constraints.

        Returns:
            NDArray[np.float64]: Vector of shape (n,) where b[i] is
            bidder i's budget constraint.
        """
        return self._b

    @b.setter
    def b(self, value: ArrayLike):
        try:
            array = np.array(value, dtype=np.float64)
            if array.shape != (self.n,):
                raise ValueError(f"b must have shape ({self.n},), got {array.shape}")
            self._b = array
        except (ValueError, TypeError):
            raise TypeError(
                f"b must be convertible to a numpy array with shape ({self.n},)"
            )

    @property
    def alpha_q(self) -> NDArray[np.uint64]:
        """
        Pacing multiplier vector representing scaled integer pacing values.

        The actual pacing multiplier used is alpha_q[i]/q for bidder i.
        Values are restricted to be positive integers not exceeding q.

        Returns:
            NDArray[np.uint64]: Vector of shape (n,) containing integer
            pacing values in range (0, q].
        """
        return self._alpha_q

    @alpha_q.setter
    def alpha_q(self, value: ArrayLike):
        try:
            # First convert to float to check values, then to uint64
            temp_array = np.array(value, dtype=np.float64)
            if temp_array.shape != (self.n,):
                raise ValueError(
                    f"alpha_q must have shape ({self.n},), got {temp_array.shape}"
                )
            if np.any(temp_array < 0):
                raise ValueError("All values in alpha_q must be positive")
            if np.any(temp_array > self.q):
                raise ValueError(f"All values in alpha_q must be <= q ({self.q})")
            self._alpha_q = temp_array.astype(np.uint64)
        except (ValueError, TypeError):
            raise TypeError(
                f"alpha_q must be convertible to a numpy array with shape ({self.n},)"
            )

    # Utility functions
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
        return np.sum(np.multiply(self._v, x), axis=1)

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
        return np.sum(np.multiply(self._v - p, x), axis=1)

    def revenue(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """
        Calculate the revenue of the auction given the allocation and prices

        Parameters:
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
            p: NDArray[np.float64]
                The prices of the auctions (m)
        Returns:
            float
            The revenue of the auction
        """
        return np.sum(p)

    def social_welfare(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """
        Calculate the social welfare of the auction given the allocation and prices

        Parameters:
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
            p: NDArray[np.float64]
                The prices of the auctions (m)
        Returns:
            float
            The social welfare of the auction
        """
        return np.sum(np.multiply(self._v, x))

    def liquid_welfare(self, x: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """
        Calculate the liquid welfare of the auction given the allocation and prices

        Parameters:
            x: NDArray[np.float64]
                The allocation for each bidder and auction (n x m)
            p: NDArray[np.float64]
                The prices of the auctions (m)
        Returns:
            float
            The liquid welfare of the auction
        """
        return np.sum(np.minimum(self._b, np.sum(np.multiply(self._v, x), axis=1)))

    def bids(self) -> NDArray[np.float64]:
        """
        Generate bids for all bidders

        Returns:
            The bids for all bidders as a NumPy matrix of shape (n, m)
        """
        return (self._v * (self._alpha_q[:, np.newaxis] / self.q)).astype(np.float64)

    def fpa(
        self, bids: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | Violation:
        """
        First Price Auction procedure, used to compute the allocation of the auction
        Returns a list of allocations or a Violation if a bidder is over budget

        Parameters:
            bids: NDArray[np.float64]
                The bids for all bidders as a NumPy matrix of shape (n, m)

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]] | Violation
            The tuple of (allocations, prices) where prices[j] is the price for item j
        """
        spending = np.zeros(self.n)
        x: NDArray[np.float64] = np.zeros((self.n, self.m))
        p: NDArray[np.float64] = np.zeros(self.m)

        for auction in range(self.m):
            winning_bidders: list[int] = []
            winning_bid = -1.0

            for bidder in range(self.n):
                bid: float = bids[bidder][auction]
                if bid > winning_bid:
                    winning_bidders = [bidder]
                    winning_bid = bid
                elif bid == winning_bid:
                    winning_bidders.append(bidder)

            assert len(winning_bidders) > 0
            for winning_bidder in winning_bidders:
                x_ij = 1 / len(winning_bidders)
                spending[winning_bidder] += winning_bid * x_ij
                if spending[winning_bidder] > self._b[winning_bidder]:
                    return Violation(winning_bidder, auction)

                x[winning_bidder][auction] = x_ij
            p[auction] = winning_bid

        return x, p

    def auction(
        self, adjustment: tuple[int, int] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the allocation of the auction given the current state

        Parameters:
            adjustment: Optional[tuple[int, int]] = None
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
            bids[bidder] = self._v[bidder] * (new_alpha_q / self.q)

        while True:
            valid_bids = np.multiply(mask, bids)
            match self.fpa(valid_bids):
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)
                case x, p:
                    return x, p

    def best_response_candidates(self, bidder: int) -> set[int]:
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
                if other_bidder == bidder or self._v[bidder][auction] == 0:
                    continue

                other_v = self._v[other_bidder][auction]
                other_alpha_q = self._alpha_q[other_bidder]
                bidder_v = self._v[bidder][auction]

                new_alpha_q: float = (other_v * other_alpha_q) / bidder_v

                # Match their bid
                if new_alpha_q.is_integer() and new_alpha_q <= self.q:
                    new_alpha_qs.add(int(new_alpha_q))

                # Outbid them
                new_alpha_q = floor(new_alpha_q) + 1
                if new_alpha_q <= self.q:
                    new_alpha_qs.add(new_alpha_q)

                # Debugging
                new_bid = bidder_v * new_alpha_q
                other_bid = other_v * other_alpha_q
                if new_bid < other_bid:
                    logger.error(f"{new_bid} < {other_bid}")

        return new_alpha_qs

    def best_response(self, bidder: int) -> BestResponse:
        """
        Find the best response for a bidder, given the current state
        If threaded is enabled, this function will use threads to calculate the best response

        Parameters:
            bidder: int
                The index of the bidder to find the best response for

        Returns:
            BestResponse
            The best response for the bidder
        """
        curr_alpha_q = self._alpha_q[bidder]
        initial_x, initial_p = self.auction()
        curr_util = self.utility(initial_x, initial_p)

        max_util: NDArray[np.float64] = curr_util
        max_alpha_q = curr_alpha_q
        new_alpha_qs = self.best_response_candidates(bidder)

        if self.threaded:
            executor = self.executor or ThreadPoolExecutor()

            def func(alpha_q: int, bidder: int) -> NDArray[np.float64]:
                x, p = self.auction((bidder, alpha_q))
                return self.utility(x, p)

            future_to_alpha_q = {
                executor.submit(func, new_alpha_q, bidder): new_alpha_q
                for new_alpha_q in new_alpha_qs
            }

            for future in as_completed(future_to_alpha_q):
                new_alpha_q = future_to_alpha_q[future]
                new_utility = future.result()
                if new_utility[bidder] > max_util[bidder] or (
                    new_utility[bidder] == max_util[bidder]
                    and new_alpha_q > max_alpha_q
                ):
                    max_util = new_utility
                    max_alpha_q = new_alpha_q
        else:
            for new_alpha_q in sorted(new_alpha_qs):
                x, p = self.auction((bidder, new_alpha_q))
                new_utility = self.utility(x, p)
                if new_utility[bidder] >= max_util[bidder]:
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
        seen = {tuple(self._alpha_q): 0}
        order = list(range(self.n))

        while True:
            old_state = tuple(self._alpha_q)

            # Shuffle the order of bidders if shuffle is enabled
            if self.shuffle and len(order) > 1:
                last_bidder = order[-1]
                np.random.shuffle(order)
                if order[0] == last_bidder:
                    swap_idx = np.random.randint(1, len(order))
                    order[0], order[swap_idx] = order[swap_idx], order[0]

            for bidder in order:
                br = self.best_response(bidder)
                d_utility = br.new_utility - br.old_utility
                if d_utility[bidder] > self.epsilon:
                    bidder_utilty = br.new_utility[bidder]
                    self._alpha_q[bidder] = br.new_alpha_q
                else:
                    bidder_utilty = br.old_utility[bidder]

                if self.stats:
                    x, p = self.auction()
                    self.stats["utility"][bidder].append(bidder_utilty)
                    self.stats["alpha_q"][bidder].append(self._alpha_q[bidder])
                    self.stats["social_welfare"].append(self.social_welfare(x, p))
                    self.stats["liquid_welfare"].append(self.liquid_welfare(x, p))
                    self.stats["revenue"].append(self.revenue(x, p))

            new_state = tuple(self._alpha_q)

            # PNE found
            if new_state == old_state:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                return PNE(
                    iteration,
                    self.alpha_q[:],
                    *self.auction(),
                    stats=deepcopy(self.stats),
                )

            # Cycle detection
            if new_state in seen:
                if self.stats:
                    self.stats["time"] = perf_counter() - self.stats["time"]
                    self.stats["cycle_length"] = iteration - seen[new_state]
                return Cycle(iteration, self.alpha_q[:], stats=deepcopy(self.stats))

            seen[new_state] = iteration
            iteration += 1
