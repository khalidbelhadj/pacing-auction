from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from functools import lru_cache
from math import floor
import threading
from time import perf_counter, time
from typing import Optional, Type
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


class Simulation:
    def __init__(
        self,
        n: int,
        m: int,
        q: int = 1000,
        elim_strategy: elimination.ElimStrategy = elimination.Subsequent,
        no_budget: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        epsilon: float = 0.0,
    ) -> None:
        self.n: int = n
        self.m: int = m
        self.q: int = q
        self.elim: elimination.ElimStrategy = elim_strategy
        self.shuffle: bool = shuffle
        self.seed: int = seed if seed is not None else int(time())
        np.random.seed(self.seed)
        self.epsilon: float = epsilon

        # budget[bidder]
        self.b: NDArray[np.float64] = np.array([np.random.random() for _ in range(n)])
        if no_budget:
            self.b = np.full(n, np.inf)

        # valuation[bidder][auction]
        self.v: NDArray[np.float64] = np.random.random((n, m))

        # alpha[bidder], multiples of q
        self.alpha_q: NDArray[np.uint64] = np.array(
            [np.random.randint(0, q) for _ in range(n)]
        )

    @classmethod
    def load(cls, file_path: str) -> "Simulation":
        with open(file_path, "r") as file:
            data = json.load(file)

        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)

        return instance

    def save(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(self.__dict__, file)

    def utility(self, bidder: int, allocations: list[Allocation]) -> float:
        utility = 0
        for a in allocations:
            if bidder in a.bidders:
                utility += self.v[bidder][a.auction] - a.price
        return utility

    def fpa(
        self, mask: NDArray[np.bool], adjust: Optional[tuple[int, float]] = None
    ) -> FPAResult:
        bids = self.v * (self.alpha_q[:, np.newaxis] / self.q)
        if adjust:
            bidder, adjustment = adjust
            bids[bidder] = self.v[bidder] * (adjustment / self.q)
        valid_bids = np.where(mask, bids, 0)
        winners: NDArray[np.int64] = np.argwhere(
            valid_bids == np.max(valid_bids, axis=0)
        )
        allocations: list[Allocation] = [
            Allocation([], auction, 0.0) for auction in range(self.m)
        ]

        # Assign winners to auctions
        for winner, auction in winners:
            allocations[auction].bidders.append(winner)

        # Calculate prices
        for i in range(len(allocations)):
            a = allocations[i]
            assert len(a.bidders) > 0
            price = valid_bids[a.bidders[0]][a.auction]
            allocations[i] = Allocation(a.bidders, a.auction, price / len(a.bidders))

        # Check for budget violations
        spending = np.zeros(self.n)
        for a in allocations:
            for bidder in a.bidders:
                spending[bidder] += a.price
                if spending[bidder] > self.b[bidder]:
                    return Violation(bidder, a.auction)

        return FPAAllocation(allocations)

    def auction(self, adjust: Optional[tuple[int, float]] = None) -> list[Allocation]:
        # mask[bidder][auction], True if not eliminated
        mask = np.ones((self.n, self.m), dtype=bool)

        while True:
            match self.fpa(mask, adjust):
                case FPAAllocation(allocations):
                    return allocations
                case Violation(bidder, auction):
                    self.elim.eliminate(bidder, auction, mask)
                case _:
                    pass

    def best_response_auction(self, bidder: int, auction: int) -> BestResponse:
        curr_alpha_q = self.alpha_q[bidder]
        auction_result = self.auction()
        curr_util = self.utility(bidder, auction_result)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        for other_bidder in range(self.n):
            if other_bidder == bidder or self.v[bidder][auction] == 0:
                continue

            other_bid = self.v[other_bidder][auction] * (
                self.alpha_q[other_bidder] / self.q
            )

            multiple = other_bid / self.v[bidder][auction]
            # Add 1 to outbid the other bidder by 1/q
            q_multiple = floor(multiple * self.q) + 1
            new_alpha_q = min(q_multiple, self.q)

            auction_result = self.auction((bidder, new_alpha_q))
            new_util = self.utility(bidder, auction_result)

            if new_util > max_util:
                max_util = new_util
                max_alpha_q = new_alpha_q

        return BestResponse(max_alpha_q, max_util, curr_util)

    def best_response_threaded(self, bidder: int) -> BestResponse:
        curr_util = self.utility(bidder, self.auction())
        curr_alpha_q = self.alpha_q[bidder]

        futures: list[Future[BestResponse]] = []
        with ThreadPoolExecutor() as executor:
            for auction in range(self.m):
                f = executor.submit(self.best_response_auction, bidder, auction)
                futures.append(f)

        # Accumulate results from each auction
        max_util = curr_util
        max_alpha_q = curr_alpha_q
        for f in as_completed(futures):
            res = f.result()
            if res.new_utility > max_util:
                max_util = res.new_utility
                max_alpha_q = res.new_alpha_q

        return BestResponse(max_alpha_q, max_util, curr_util)

    def best_response(self, bidder: int) -> BestResponse:
        curr_alpha_q = self.alpha_q[bidder]
        auction_result = self.auction()
        curr_util = self.utility(bidder, auction_result)

        max_alpha_q = curr_alpha_q
        max_util = curr_util

        for auction in range(self.m):
            for other_bidder in range(self.n):
                if other_bidder == bidder or self.v[bidder][auction] == 0:
                    continue

                other_bid = self.v[other_bidder][auction] * (
                    self.alpha_q[other_bidder] / self.q
                )

                multiple = other_bid / self.v[bidder][auction]
                # Add 1 to outbid the other bidder by 1/q
                q_multiple = floor(multiple * self.q) + 1
                new_alpha_q = min(q_multiple, self.q)

                auction_result = self.auction((bidder, new_alpha_q))
                new_util = self.utility(bidder, auction_result)

                if new_util > max_util:
                    max_util = new_util
                    max_alpha_q = new_alpha_q

        return BestResponse(max_alpha_q, max_util, curr_util)

    def welfare(self, allocations: list[Allocation]) -> float:
        return sum(self.utility(bidder, allocations) for bidder in range(self.n))

    def run(self) -> SimulationResult:
        seen = set([tuple(self.alpha_q)])
        order = np.arange(self.n)

        import matplotlib.pyplot as plt

        start_time = perf_counter()

        stats = dict()
        stats["util"] = [[] for _ in range(self.n)]

        i = 0
        while True:
            if self.shuffle:
                np.random.shuffle(order)
            utility_change = False

            for bidder in order:
                res = self.best_response(int(bidder))
                if res.new_utility > res.old_utility + self.epsilon:
                    stats["util"][bidder].append(res.new_utility - res.old_utility)
                    utility_change = True
                    self.alpha_q[bidder] = res.new_alpha_q
                else:
                    stats["util"][bidder].append(0)

            # PNE found
            if not utility_change:
                return PNE(perf_counter() - start_time, i, self.auction(), stats=stats)

            # Cycle detection
            t = tuple(self.alpha_q)
            if t in seen:
                return Cycle(perf_counter() - start_time, i, stats=stats)
            seen.add(t)

            i += 1
