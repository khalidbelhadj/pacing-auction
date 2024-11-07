import collections
from math import ceil
from random import random, randint
import sys
from typing import NamedTuple, Optional
import json

from elimination import All, Elimination


class Allocation(NamedTuple):
    """
    An allocation of the auction, signifying the winner
    """

    bidder: int
    auction: int
    price: float

    def __repr__(self) -> str:
        return f"Allocation(bidder={self.bidder}, auction={self.auction}, price={self.price})"


MAX_STEPS = 10 * 10**2


class Simulation:
    def __init__(
        self,
        n: int = 10,
        m: int = 3,
        q: int = 100,
        elimination: Elimination = All(),
        file: Optional[str] = None,
        check_cycle: bool = False,
    ) -> None:
        """
        Initialize the simulation with given parameters.

        :param n: Number of bidders
        :param m: Number of auctions
        :param q: Quality factor
        :param elimination: Elimination strategy
        """
        self.elimination = elimination
        self.check_cycle = check_cycle

        if file:
            with open(file) as f:
                state = json.load(f)
                try:
                    self.n = state["n"]
                    self.m = state["m"]
                    self.q = state["q"]
                    self.budget = state["budget"]
                    self.valuation = state["valuation"]
                    self.alpha = state["alpha"]
                except KeyError:
                    raise ValueError("Invalid state file")
        else:
            self.n: int = n
            self.m: int = m
            self.q: int = q

            # budget[bidder]
            self.budget = [random() for _ in range(n)]

            # valuation[auction][bidder]
            self.valuation = [[random() for _ in range(n)] for _ in range(m)]

            # alpha[bidder]
            self.alpha = [randint(0, q) / q for _ in range(n)]

    def eliminate(self, bidder: int, auction: int) -> None:
        """
        Eliminate the violation.

        :param violation: The violation to eliminate
        """
        self.elimination.eliminate(bidder, auction)

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        """
        Check if the bidder is eliminated.

        :param bidder: The bidder to check
        :param auction: The auction to check
        :return: True if the bidder is eliminated, False otherwise
        """
        return self.elimination.is_eliminated(bidder, auction)

    def run_fpa(self) -> list[Allocation]:
        """
        Run the first price auction.

        :return: list of allocations
        """
        allocations: list[Allocation] = []

        for auction in range(self.m):
            winner = None
            winning_bid = -1

            for bidder in range(self.n):
                if self.is_eliminated(bidder, auction):
                    continue

                alpha = self.alpha[bidder]
                bid = self.valuation[auction][bidder] * alpha
                if winner is None or bid > winning_bid:
                    winner = bidder
                    winning_bid = bid

            if winner is not None:
                allocations.append(Allocation(winner, auction, winning_bid))

        return allocations

    def check_violations(
        self, allocations: list[Allocation]
    ) -> Optional[tuple[int, int]]:
        """
        Check if there are any violations, and return the first violation found.

        :param allocations: list of allocations
        :return: The first violation found, or None if no violations
        """
        spending: dict[int, float] = collections.defaultdict(float)

        for bidder, auction, price in allocations:
            spending[bidder] += price
            if spending[bidder] > self.budget[bidder]:
                return (bidder, auction)

        return None

    def step(self) -> list[Allocation]:
        """
        Run a step of the simulation.

        :return: list of allocations
        """
        allocations = self.run_fpa()
        violation = self.check_violations(allocations)
        if violation is not None:
            self.eliminate(*violation)
            return self.step()
        return allocations

    def utility(self, bidder: int, allocations: list[Allocation]) -> float:
        """
        Calculate the utility of a bidder given the allocations
        utility = budget - spending
        """
        spending = sum(price for winner, _, price in allocations if winner == bidder)
        return self.budget[bidder] - spending

    def best_response(self, bidder: int) -> None:
        """
        Finds the best response for a bidder
        """
        current_utility = self.utility(bidder, self.run_fpa())
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
                new_alpha = min(1, ceil(new_alpha * self.q) / self.q)
                self.alpha[bidder] = new_alpha
                utility = self.utility(bidder, self.run_fpa())
                if utility > max_utility:
                    max_utility = utility
                    max_alpha = self.alpha[bidder]

        if max_utility > current_utility:
            self.alpha[bidder] = max_alpha

    def dump(self) -> None:
        """
        Dump the current state to a JSON file
        """

        with open("state.json", "w") as f:
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

    def run(self) -> list[Allocation]:
        """
        Run the simulation
        """

        self.dump()

        seen = set([tuple(self.alpha)])
        for i in range(MAX_STEPS):
            utility_change = False
            # Best response for each bidder sequentially
            for bidder in range(self.n):
                current_utility = self.utility(bidder, self.run_fpa())
                self.best_response(bidder)
                new_utility = self.utility(bidder, self.run_fpa())
                utility_change = utility_change or new_utility > current_utility

            if self.check_cycle:
                if tuple(self.alpha) in seen:
                    print("Cycle", i)
                    break
                seen.add(tuple(self.alpha))

            if not utility_change:
                print("PNE found at iteration", i)
                break
        else:
            print(f"PNE not found after {MAX_STEPS:,} iterations", file=sys.stderr)
        return self.step()
