import collections
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import NamedTuple, Optional
import json

import numpy as np
from numpy.random import shuffle, random, randint  # type: ignore
from numpy.typing import NDArray

from allocation import Allocation
from elimination import Elimination, Subsequent


@dataclass
class SimulationResult:
    time: float


@dataclass
class Cycle(SimulationResult):
    iteration: int


@dataclass
class PNE(SimulationResult):
    allocations: list[Allocation]
    iteration: int


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
        elimination: Optional[Elimination] = None,
        no_budget: bool = False,
    ) -> None:
        self.elimination: Elimination = elimination or Subsequent(n, m)

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

    def reset(self) -> None:
        self.elimination.clear()
        self.__init__(self.n, self.m, self.q, self.elimination)

    def load(self, file: str) -> None:
        raise NotImplementedError
        # with open(file) as f:
        #     state = json.load(f)
        #     try:
        #         self.n = state["n"]
        #         self.m = state["m"]
        #         self.q = state["q"]
        #         self.b = np.array(state["budget"])
        #         # assert self.b.size == self.n, "Invalid budget, must have n elements"
        #         self.v = np.array([np.array(vi) for vi in state["valuation"]])
        #         # assert (
        #         #     self.v.size == self.n
        #         # ), "Invalid valuation, must have n x m elements"
        #         self.alpha = np.array(state["alpha"])
        #         # assert len(self.alpha) == self.n, "Invalid alpha, must have n elements"
        #     except KeyError:
        #         raise ValueError("Invalid state file")

    def save(self, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(
                {
                    "n": self.n,
                    "m": self.m,
                    "q": self.q,
                    "budget": self.b,
                    "valuation": self.v,
                    "alpha": self.alpha,
                },
                f,
                indent=4,
            )

    def fpa(self) -> FPAAllocation | Violation:
        allocations: list[Allocation] = []
        spending: dict[int, float] = collections.defaultdict(float)
        bids = self.v * self.alpha[:, np.newaxis]

        for auction in range(self.m):
            winner = None
            winning_bid = -1

            for bidder in range(self.n):
                bid = bids[bidder][auction]

                if bid > winning_bid and not self.elimination.is_eliminated(
                    bidder, auction
                ):
                    winner = bidder
                    winning_bid = bid

            if winner is not None:
                allocations.append(Allocation(winner, auction, winning_bid))
                spending[winner] += winning_bid
                if spending[winner] > self.b[winner]:
                    return Violation(winner, auction)

        return FPAAllocation(allocations)

    def auction(self) -> list[Allocation]:
        self.elimination.clear()
        while True:
            match self.fpa():
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    self.elimination.eliminate(bidder, auction)

    def utility(self, bidder: int, allocations: list[Allocation]) -> float:
        utility = 0
        for winner, auction, price in allocations:
            if winner == bidder:
                utility += self.v[bidder][auction] - price
        return utility

    def best_response(self, bidder: int) -> bool:
        current_utility = self.utility(bidder, self.auction())
        max_utility = current_utility
        max_alpha = self.alpha[bidder]

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder or self.v[bidder][auction] == 0:
                    continue

                # Calculate utility, if bidder matches the other bidder's bid
                other_bid = self.alpha[other_bidder] * self.v[other_bidder][auction]
                new_alpha = other_bid / self.v[bidder][auction]
                self.alpha[bidder] = min(ceil(new_alpha * self.q) / self.q, 1)
                utility = self.utility(bidder, self.auction())

                if utility > max_utility:
                    # return True
                    max_utility = utility
                    max_alpha = self.alpha[bidder]

        # return False
        self.alpha[bidder] = max_alpha
        return max_utility > current_utility

    def welfare(self, allocations: list[Allocation]) -> float:
        return sum(self.utility(bidder, allocations) for bidder in range(self.n))

    def run(self) -> SimulationResult:
        seen = set([tuple(self.alpha)])
        order = np.arange(self.n)

        start_time = perf_counter()

        i = 0
        while True:
            # shuffle(order)
            # print(i)
            utility_change = False

            for bidder in order:
                utility_change = self.best_response(bidder) or utility_change

            # PNE found
            if not utility_change:
                return PNE(perf_counter() - start_time, self.auction(), i)

            # Cycle detection
            t = tuple(self.alpha)
            if t in seen:
                return Cycle(perf_counter() - start_time, i)
            seen.add(t)

            i += 1
