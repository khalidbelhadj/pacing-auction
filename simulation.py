import collections
from math import ceil
from random import random, randint, shuffle
from typing import NamedTuple
import json

from allocation import Allocation
from elimination import Elimination, Subsequent


class Cycle(NamedTuple):
    iteration: int


class PNE(NamedTuple):
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
        elimination: Elimination = Subsequent(),
        no_budget: bool = False,
    ) -> None:
        """
        Initialize the simulation with given parameters.

        :param n: Number of bidders
        :param m: Number of auctions
        :param q: Quality factor
        :param elimination: Elimination strategy (default Subsequent)
        :param no_budget: Whether to remove all budgets
        """
        self.elimination: Elimination = elimination

        self.n: int = n
        self.m: int = m
        self.q: int = q

        # budget[bidder]
        self.budget: list[float] = [random() for _ in range(n)]
        if no_budget:
            self.budget = [float("infinity")] * n

        # valuation[auction][bidder]
        self.valuation: list[list[float]] = [
            [random() for _ in range(n)] for _ in range(m)
        ]

        # alpha[bidder]
        self.alpha: list[float] = [randint(0, q) / q for _ in range(n)]

    def load(self, file: str) -> None:
        with open(file) as f:
            state = json.load(f)
            try:
                self.n = state["n"]
                self.m = state["m"]
                self.q = state["q"]
                self.budget = state["budget"]
                assert (
                    len(self.budget) == self.n
                ), "Invalid budget, must have n elements"
                self.valuation = state["valuation"]
                assert len(self.valuation) == self.m and (
                    self.m == 0 or len(self.valuation[0]) == self.n
                ), "Invalid valuation, must have m x n elements"
                self.alpha = state["alpha"]
                assert len(self.alpha) == self.n, "Invalid alpha, must have n elements"
            except KeyError:
                raise ValueError("Invalid state file")

    def save(self, file_name: str) -> None:
        """
        Dump the current state to a JSON file.

        :param file_name: The name of the file to save the state to
        """
        with open(file_name, "w") as f:
            json.dump(
                {
                    "n": self.n,
                    "m": self.m,
                    "q": self.q,
                    "budget": self.budget,
                    "valuation": self.valuation,
                    "alpha": self.alpha,
                },
                f,
                indent=4,
            )

    def fpa(self) -> FPAAllocation | Violation:
        """
        Run the first price auction.

        :return: List of allocations
        """
        allocations: list[Allocation] = []
        spending: dict[int, float] = collections.defaultdict(float)

        for auction in range(self.m):
            winner = None
            winning_bid = -1

            for bidder in range(self.n):
                if self.elimination.is_eliminated(bidder, auction):
                    continue

                alpha = self.alpha[bidder]
                bid = self.valuation[auction][bidder] * alpha
                if winner is None or bid > winning_bid:
                    winner = bidder
                    winning_bid = bid

            if winner is not None:
                allocations.append(Allocation(winner, auction, winning_bid))
                spending[winner] += winning_bid
                if spending[winner] > self.budget[winner]:
                    return Violation(winner, auction)

        return FPAAllocation(allocations)

    def auction(self) -> list[Allocation]:
        """
        Run a step of the simulation.

        :return: List of allocations
        """
        self.elimination.clear()
        while True:
            match self.fpa():
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    self.elimination.eliminate(bidder, auction)

    def utility(self, bidder: int, allocations: list[Allocation]) -> float:
        """
        Calculate the utility of a bidder given the allocations.

        :param bidder: The bidder whose utility is being calculated
        :param allocations: List of allocations
        :return: The utility of the bidder
        """
        utility = 0
        for winner, auction, price in allocations:
            if winner == bidder:
                utility += self.valuation[auction][bidder] - price
        return utility

    def best_response(self, bidder: int) -> bool:
        """
        Finds the best response for a bidder.

        :param bidder: The bidder whose best response is being calculated
        :return: True if the utility of the bidder has changed, False otherwise
        """
        current_utility = self.utility(bidder, self.auction())
        max_utility = current_utility
        max_alpha = self.alpha[bidder]

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder:
                    continue

                other_bid = (
                    self.alpha[other_bidder] * self.valuation[auction][other_bidder]
                )

                new_alpha = other_bid / self.valuation[auction][bidder]
                self.alpha[bidder] = min(ceil(new_alpha * self.q) / self.q, 1)

                utility = self.utility(bidder, self.auction())

                if utility > max_utility:
                    max_utility = utility
                    max_alpha = self.alpha[bidder]

        self.alpha[bidder] = max_alpha
        return max_utility > current_utility

    def run(self) -> Cycle | PNE:
        """
        Run the simulation.

        :return: List of allocations
        """
        seen = set([tuple(self.alpha)])
        order = list(range(self.n))

        i = 0
        while True:
            shuffle(order)
            print(i, order)
            utility_change = False

            for bidder in order:
                utility_change = self.best_response(bidder) or utility_change

            # PNE found
            if not utility_change:
                return PNE(self.auction(), i)

            # Cycle detection
            t = tuple(self.alpha)
            if t in seen:
                return Cycle(i)
            seen.add(t)

            i += 1
