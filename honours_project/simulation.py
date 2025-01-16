from dataclasses import dataclass
from math import floor
from time import perf_counter, time
from typing import Optional
import json


import numpy as np
from numpy.typing import NDArray

from honours_project.data import (
    SimulationResult,
    PNE,
    Cycle,
    Allocation,
    FPAResult,
    Violation,
    FPAAllocation,
    BestResponse,
)
import honours_project.elimination as elimination

import logging

logger = logging.getLogger("simulation")


@dataclass
class Simulation:
    def __init__(
        self,
        n: int,
        m: int,
        q: int = 1000,
        no_budget: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.seed = seed if seed is not None else int(time())
        np.random.seed(self.seed)

        self.n: int = n
        self.m: int = m
        self.q: int = q
        self.shuffle: bool = shuffle

        # budget[bidder]
        self.b: NDArray[np.float64] = np.array([np.random.random() for _ in range(n)])
        if no_budget:
            self.b = np.full(n, np.inf)

        # valuation[bidder][auction]
        self.v: NDArray[np.float64] = np.random.random((n, m))

        # alpha[bidder], multiples of q
        self.alpha: NDArray[np.float64] = np.array(
            [np.random.randint(0, q) / q for _ in range(n)]
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
        for a in allocations:
            if a.bidder == bidder:
                utility += self.v[bidder][a.auction] - a.price
        return utility

    def fpa(
        self, mask: NDArray[np.bool_], adjust: Optional[tuple[int, float]] = None
    ) -> FPAResult:
        bids = self.v * self.alpha[:, np.newaxis]
        if adjust:
            bidder, adjustment = adjust
            bids[bidder] = self.v[bidder] * adjustment
        valid_bids = np.where(mask, bids, -1)

        winners: NDArray[np.int_] = np.argmax(valid_bids, axis=0)
        spending = np.zeros(self.n)
        allocations: list[Allocation] = [Allocation(-1, -1, -1)] * self.m

        for auction, winner in enumerate(winners):
            bid = valid_bids[winner][auction]
            if spending[winner] + bid > self.b[winner]:
                return Violation(winner, auction)

            spending[winner] += bid
            allocations[auction] = Allocation(winner, auction, max(0, bid))

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
                case _:
                    pass

    # def best_response_threaded(self, bidder: int) -> bool:
    #     util = self.utility(bidder, self.auction())

    #     def compute(auction: int, other_bidder: int) -> tuple[float, float]:
    #         # Calculate utility, if bidder matches the other bidder's bid
    #         other_bid = self.alpha[other_bidder] * self.v[other_bidder][auction]
    #         new_alpha = other_bid / self.v[bidder][auction]
    #         new_alpha = min(ceil(new_alpha * self.q) / self.q, 1)
    #         new_util = self.utility(bidder, self.auction((bidder, new_alpha)))
    #         return (new_util, new_alpha)

    #     futures: list[Future[tuple[float, float]]] = []
    #     with ThreadPoolExecutor() as executor:
    #         for auction in range(self.m):
    #             for other_bidder in range(self.n):
    #                 if other_bidder == bidder or self.v[bidder][auction] == 0:
    #                     continue

    #                 f = executor.submit(compute, auction, other_bidder)
    #                 futures.append(f)

    #     result = [f.result() for f in futures]
    #     result.append((util, self.alpha[bidder]))
    #     max_util, max_alpha = max(result)
    #     self.alpha[bidder] = max_alpha
    #     return max_util > util

    def bids(self):
        return self.v * self.alpha[:, np.newaxis]

    def best_response(self, bidder: int) -> BestResponse:
        curr_alpha = self.alpha[bidder]
        auction_result = self.auction()
        curr_util = self.utility(bidder, auction_result)

        max_alpha = curr_alpha
        max_util = curr_util

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder or self.v[bidder][auction] == 0:
                    continue

                other_bid = self.alpha[other_bidder] * self.v[other_bidder][auction]

                multiple = other_bid / self.v[bidder][auction]
                # Add 1 to outbid the other bidder by 1/q
                q_multiple = floor(multiple * self.q) + 1
                new_alpha = min(q_multiple / self.q, 1.0)

                auction_result = self.auction((bidder, new_alpha))
                new_util = self.utility(bidder, auction_result)

                if new_util > max_util:
                    max_util = new_util
                    max_alpha = new_alpha

        return BestResponse(max_alpha, max_util, curr_util)

    def welfare(self, allocations: list[Allocation]) -> float:
        return sum(self.utility(bidder, allocations) for bidder in range(self.n))

    def run(self) -> SimulationResult:
        seen = set([tuple(self.alpha)])
        order = np.arange(self.n)

        start_time = perf_counter()

        i = 0
        while True:
            if self.shuffle:
                np.random.shuffle(order)
            utility_change = False

            for bidder in order:
                res = self.best_response(int(bidder))
                if res.new_utility > res.old_utility:
                    utility_change = True
                    # round new alpha to 1/q
                    self.alpha[bidder] = res.new_alpha * self.q

            # PNE found
            if not utility_change:
                return PNE(perf_counter() - start_time, i, self.auction())

            # Cycle detection
            t = tuple(self.alpha)
            if t in seen:
                return Cycle(perf_counter() - start_time, i)
            seen.add(t)

            i += 1
