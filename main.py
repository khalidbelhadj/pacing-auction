from collections import namedtuple
import collections
from random import randint
from typing import Callable, NamedTuple, Optional
from attr import dataclass
from cycler import K
from numpy import argmax
from numpy.random import randint, normal
from scipy.stats import norm


class Violation(NamedTuple):
    bidder: int
    auction: int


type Allocations = list[tuple[int, float]]


def f():
    raise NotImplementedError()


class Simulation:
    def __init__(self, n=10, m=3, q=10) -> None:
        self.n: int = n
        self.m: int = m
        self.q: int = 10

        budget_mu = 100
        budget_sd = 2
        self.budget = [normal(budget_mu, budget_sd) for _ in range(n)]

        # valuation of each auction, for each bidder: valuation[auction][bidder]
        self.valuation = [[] for _ in range(m)]

        for i in range(n):
            budget = self.budget[i]
            val_sd = budget_sd
            val_mu = budget / m
            for j in range(m):
                self.valuation[j].append(normal(val_mu, val_sd))

        # pacing multiplier for each bidder
        self.alpha = [q / q for _ in range(n)]

    def is_feasible(self) -> bool:
        """
        Check if the current self is feasible
        """
        return False

    def run_fpa(self) -> Allocations:
        """
        Run the first price auction
        """
        allocations = []

        for j in range(self.m):
            bids = [self.valuation[j][i] * self.alpha[i] for i in range(self.n)]
            winner = argmax(bids)
            winning_bid = bids[winner]
            allocations.append((winner, winning_bid))

        return allocations

    def check_violations(self, allocations: Allocations) -> Optional[Violation]:
        """
        Check if there are any violations, and return the first violation found
        """
        spending = collections.defaultdict(float)
        for j in range(self.m):
            winner, price = allocations[j]
            spending[winner] += price
            if spending[winner] > self.budget[winner]:
                return Violation(winner, j)
        return None

    def eliminate(self, allocations: Allocations, violation: Violation):
        """
        Eliminate the violation
        """
        exit(1)
        pass

    def run(self) -> Allocations:
        """
        Run the simulation
        """
        while True:
            allocations = self.run_fpa()
            violation = self.check_violations(allocations)
            if violation is None:
                return allocations
            else:
                self.eliminate(allocations, violation)


def main() -> None:
    self = Simulation(n=4, m=3)
    allocations = self.run()
    print(allocations)


if __name__ == "__main__":
    main()
