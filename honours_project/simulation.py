from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import NamedTuple, Optional
import json


import numpy as np
from numpy.random import shuffle, random, randint  # type: ignore
from numpy.typing import NDArray

from honours_project.allocation import Allocation
import honours_project.elimination as elimination

from concurrent.futures import Future, ThreadPoolExecutor

import logging

logger = logging.getLogger("simulation")


@dataclass
class SimulationResult:
    time: float
    iteration: int


@dataclass
class Cycle(SimulationResult):
    pass


@dataclass
class PNE(SimulationResult):
    allocations: list[Allocation]


class Violation(NamedTuple):
    bidder: int
    auction: int


class FPAAllocation(NamedTuple):
    allocations: list[Allocation]


class Simulation:
    def __init__(
        self,
        n: int,
        m: int,
        q: int,
        no_budget: bool = False,
    ) -> None:

        self.n: int = n
        self.m: int = m
        self.q: int = q

        # budget[bidder]
        self.b: NDArray[np.float64] = np.array([random() for _ in range(n)])
        if no_budget:
            self.b = np.full(n, np.inf)

        # valuation[bidder][auction]
        self.v: NDArray[np.float64] = random((n, m))

        # alpha[bidder]
        self.alpha: NDArray[np.float64] = np.array(
            [randint(0, q) / q for _ in range(n)]
        )

    def load(self, file: str) -> None:
        with open(file) as f:
            state = json.load(f)
            try:
                self.n = state["n"]
                self.m = state["m"]
                self.q = state["q"]
                self.b = np.array(state["budget"])
                self.v = np.array([np.array(vi) for vi in state["valuation"]])
                self.alpha = np.array(state["alpha"])
            except KeyError:
                raise ValueError("Invalid state file")

    def save(self, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(
                {
                    "n": self.n,
                    "m": self.m,
                    "q": self.q,
                    "budget": list(self.b),
                    "valuation": [list(v) for v in self.v],
                    "alpha": list(self.alpha),
                },
                f,
                indent=4,
            )

    def utility(self, bidder: int, allocations: list[Allocation]) -> float:
        utility = 0
        for winner, auction, price in allocations:
            if winner == bidder:
                utility += self.v[bidder][auction] - price
        return utility

    def fpa(
        self, mask: NDArray[np.bool_], adjust: Optional[tuple[int, float]] = None
    ) -> FPAAllocation | Violation:
        bids = self.v * self.alpha[:, np.newaxis]
        if adjust:
            bidder, adjustment = adjust
            bids[bidder] = self.v[bidder] * adjustment
        valid_bids = np.where(mask, bids, 0)

        winners: NDArray[np.int_] = np.argmax(valid_bids, axis=0)
        spending = np.zeros(self.n)
        allocations: list[Allocation] = []

        for auction, winner in enumerate(winners):
            bid = valid_bids[winner][auction]
            if spending[winner] + bid > self.b[winner]:
                return Violation(winner, auction)

            spending[winner] += bid
            allocations.append(Allocation(int(winner), int(auction), float(bid)))

        assert len(allocations) == self.m

        return FPAAllocation(allocations)

    def auction(self, adjust: Optional[tuple[int, float]] = None) -> list[Allocation]:
        # mask[bidder][auction], True if not eliminated
        mask = np.ones((self.n, self.m), dtype=bool)

        while True:
            match self.fpa(mask, adjust):
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    elimination.subsequent(bidder, auction, mask)

    def best_response_threaded(self, bidder: int) -> bool:
        util = self.utility(bidder, self.auction())

        def compute(auction: int, other_bidder: int) -> tuple[float, float]:
            # Calculate utility, if bidder matches the other bidder's bid
            other_bid = self.alpha[other_bidder] * self.v[other_bidder][auction]
            new_alpha = other_bid / self.v[bidder][auction]
            new_alpha = min(ceil(new_alpha * self.q) / self.q, 1)
            new_util = self.utility(bidder, self.auction((bidder, new_alpha)))
            return (new_util, new_alpha)

        futures: list[Future[tuple[float, float]]] = []
        with ThreadPoolExecutor() as executor:
            for auction in range(self.m):
                for other_bidder in range(self.n):
                    if other_bidder == bidder or self.v[bidder][auction] == 0:
                        continue

                    f = executor.submit(compute, auction, other_bidder)
                    futures.append(f)

        result = [f.result() for f in futures]
        result.append((util, self.alpha[bidder]))
        max_util, max_alpha = max(result)
        self.alpha[bidder] = max_alpha
        return max_util > util

    def best_response(self, bidder: int) -> bool:
        curr_alpha = self.alpha[bidder]
        curr_util = self.utility(bidder, self.auction())

        max_alpha = curr_alpha
        max_util = curr_util

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder or self.v[bidder][auction] == 0:
                    continue

                other_bid = self.alpha[other_bidder] * self.v[other_bidder][auction]
                new_alpha = other_bid / self.v[bidder][auction]
                new_alpha = min(ceil(new_alpha * self.q) / self.q, 1)
                new_util = self.utility(bidder, self.auction((bidder, new_alpha)))

                if new_util > max_util:
                    max_util = new_util
                    max_alpha = new_alpha

        self.alpha[bidder] = max_alpha
        return max_util > curr_util

    def welfare(self, allocations: list[Allocation]) -> float:
        return sum(self.utility(bidder, allocations) for bidder in range(self.n))

    def run(self) -> SimulationResult:
        seen = set([tuple(self.alpha)])
        order = np.arange(self.n)

        start_time = perf_counter()

        i = 0
        while True:
            # shuffle(order)
            utility_change = False

            for bidder in order:
                # TODO: alpha should be set here, and not mutated anywhere else
                utility_change = (
                    # self.best_response_threaded(int(bidder)) or utility_change
                    self.best_response(int(bidder))
                    or utility_change
                )

            # PNE found
            if not utility_change:
                return PNE(perf_counter() - start_time, i, self.auction())

            # Cycle detection
            t = tuple(self.alpha)
            if t in seen:
                return Cycle(perf_counter() - start_time, i)
            seen.add(t)

            i += 1
