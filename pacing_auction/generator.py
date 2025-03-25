from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from scipy import stats


@runtime_checkable
class AuctionGenerator(Protocol):
    """
    Protocol for auction generators that initialize auction instances
    with different valuation and budget patterns.
    """

    from pacing_auction import auction

    def generate(self, auction: auction.Auction) -> None: ...


class CompleteAuctionGenerator(AuctionGenerator):
    """
    Generates a complete auction where every bidder has
    a non-zero valuation for every item.

    Valuations are drawn uniformly from [0,1].
    Budgets are drawn uniformly from [0, sum of bidder's valuations].
    """

    from pacing_auction import auction

    def generate(self, auction: auction.Auction) -> None:
        n, m = auction.n, auction.m

        # Generate uniformly random valuations
        v = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                v[i, j] = np.random.uniform(0, 1)

        # Generate budgets
        b = np.zeros(n)
        v_sum = np.sum(v, axis=1)
        for i in range(n):
            b[i] = np.random.uniform(0, v_sum[i]) if v_sum[i] > 0 else 0

        auction.v = v
        auction.b = b


class SampledAuctionGenerator(AuctionGenerator):
    """
    Generates an auction where each item interests only a subset of bidders.

    For each item j, a random subset of bidders is selected to be interested.
    Interested bidders have valuations drawn uniformly from [0,1].
    Non-interested bidders have valuations of 0.

    Every bidder is guaranteed to be interested in at least one item.
    Budgets are drawn uniformly from [0, sum of bidder's valuations].
    """

    from pacing_auction import auction

    def generate(self, auction: auction.Auction) -> None:
        n, m = auction.n, auction.m

        # Choose a random subset of bidders for each auction
        interested: NDArray[np.bool_] = np.zeros((n, m), dtype=bool)
        for j in range(m):
            subset_size = np.random.randint(1, n + 1)
            subset = np.random.choice(n, size=subset_size, replace=False)
            interested[subset, j] = True

        # Ensure every bidder is interested in at least one item
        for i in range(n):
            if not np.any(interested[i]):
                j = np.random.randint(0, m)
                interested[i, j] = True

        # Generate uniformly random valuations
        v = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if interested[i, j]:
                    v[i, j] = np.random.uniform(0, 1)

        # Generate budgets
        b = np.zeros(n)
        v_sum = np.sum(v, axis=1)
        for i in range(n):
            # Ensure minimum budget is non-zero to avoid edge cases
            min_budget = 0.01 if v_sum[i] > 0 else 0
            b[i] = np.random.uniform(min_budget, max(min_budget, v_sum[i]))

        auction.v = v
        auction.b = b


class CorrelatedAuctionGenerator(AuctionGenerator):
    """
    Generates an auction with correlated valuations among bidders.

    For each item j, a random mean μ_j is drawn from [0,1].
    For each interested bidder-item pair, valuation is drawn from
    a mixture of two Gaussians with mean μ_j, standard deviation
    sigma and deviation delta.

    Parameters:
        sigma: float = 0.1
            Standard deviation for the Gaussian distribution
        delta: float = 0
            Difference between mean values of the two gaussians
    """

    from pacing_auction import auction

    def __init__(self, sigma: float = 0.1, delta: float = 0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma
        self.delta = np.clip(delta, 0, 1)

    def sample(self, mu: float) -> float:
        """
        Sample from a mixture of two truncated normals with means
        centered at mu ± delta.

        The distribution is truncated to ensure values stay within [0,1].
        """
        comp_mu: float = mu + np.random.choice([-self.delta, self.delta])

        # Ensure comp_mu is within a reasonable range to avoid numerical issues

        # Calculate the bounds for truncation
        a = (0 - comp_mu) / self.sigma
        b = (1 - comp_mu) / self.sigma

        # Sample from the truncated normal distribution
        return stats.truncnorm.rvs(a, b, loc=comp_mu, scale=self.sigma)  # type: ignore

    def generate(self, auction: auction.Auction) -> None:
        n, m = auction.n, auction.m

        # Choose a random subset of bidders for each auction
        interested: NDArray[np.bool_] = np.zeros((n, m), dtype=bool)
        for j in range(m):
            subset_size = np.random.randint(1, n + 1)
            subset = np.random.choice(n, size=subset_size, replace=False)
            interested[subset, j] = True

        # Ensure every bidder is interested in at least one item
        for i in range(n):
            if not np.any(interested[i]):
                j = np.random.randint(0, m)
                interested[i, j] = True

        # Generate correlated valuations
        v = np.zeros((n, m))
        mu = np.random.uniform(0, 1, size=m)  # Item-specific mean values
        for i in range(n):
            for j in range(m):
                if interested[i, j]:
                    v[i, j] = self.sample(mu[j])

        # Generate budgets
        b = np.zeros(n)
        v_sum = np.sum(v, axis=1)
        for i in range(n):
            # Ensure minimum budget is non-zero to avoid edge cases
            min_budget = 0.01 if v_sum[i] > 0 else 0
            b[i] = np.random.uniform(min_budget, max(min_budget, v_sum[i]))

        auction.v = v
        auction.b = b
