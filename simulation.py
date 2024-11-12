import collections
import copy
from math import ceil
from random import random, randint
import sys
from typing import NamedTuple, Optional
import json

from elimination import Elimination, Subsequent


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
        m: int = 10,
        q: int = 100,
        elimination: Elimination = Subsequent(),
        file: Optional[str] = None,
        check_cycle: bool = True,
    ) -> None:
        """
        Initialize the simulation with given parameters.

        :param n: Number of bidders
        :param m: Number of auctions
        :param q: Quality factor
        :param elimination: Elimination strategy (default Subsequent)
        :param file: Load state from file
        :param check_cycle: Check for cycles in the simulation
        """
        self.elimination = elimination
        self.check_cycle = check_cycle

        if file:
            self.load(file)
            return

        self.n: int = n
        self.m: int = m
        self.q: int = q

        # budget[bidder]
        self.budget = [random() for _ in range(n)]

        # valuation[auction][bidder]
        self.valuation = [[random() for _ in range(n)] for _ in range(m)]

        # alpha[bidder]
        self.alpha = [randint(0, q) / q for _ in range(n)]

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

    def eliminate(self, bidder: int, auction: int) -> None:
        """
        Eliminate the bidder from the auction.

        :param bidder: The bidder to eliminate
        :param auction: The auction from which the bidder is eliminated
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

        :return: List of allocations
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

        :param allocations: List of allocations
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

        :return: List of allocations
        """
        allocations = self.run_fpa()
        violation = self.check_violations(allocations)
        if violation is not None:
            self.eliminate(*violation)
            return self.step()
        return allocations

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
        current_utility = self.utility(bidder, self.step())
        max_utility = current_utility
        max_alpha = self.alpha[bidder]
        max_elim = copy.deepcopy(self.elimination)

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder:
                    continue

                other_bid = (
                    self.alpha[other_bidder] * self.valuation[auction][other_bidder]
                )

                new_alpha = other_bid / self.valuation[auction][bidder]
                self.alpha[bidder] = min(ceil(new_alpha * self.q) / self.q, 1)
                old_elim = copy.deepcopy(self.elimination)

                utility = self.utility(bidder, self.step())

                #

                if utility > max_utility:
                    max_utility = utility
                    max_alpha = self.alpha[bidder]
                    max_elim = old_elim

                self.elimination = old_elim

        self.alpha[bidder] = max_alpha
        self.elimination = max_elim
        return max_utility > current_utility

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

    def run(self) -> list[Allocation]:
        """
        Run the simulation.

        :return: List of allocations
        """
        seen = set([tuple(self.alpha)])
        for i in range(MAX_STEPS):
            utility_change = False

            for bidder in range(self.n):
                utility_change = self.best_response(bidder) or utility_change

            if not utility_change:
                print("PNE found at iteration", i)
                break

            if self.check_cycle:
                if tuple(self.alpha) in seen:
                    print("Cycle", i)
                    break
                seen.add(tuple(self.alpha))

        else:
            print(f"PNE not found after {MAX_STEPS:,} iterations", file=sys.stderr)
        return self.step()
