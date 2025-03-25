import numpy as np
from pacing_auction.data import PNE, Cycle
from pacing_auction.auction import Auction
import logging

logger = logging.getLogger("test_correctness")


def test_dominant_player():
    """
    Single dominant player with the highest budget and valuation for all auctions
    """
    sim = Auction(5, 5)
    sim.q = 1000
    sim.v = np.full((5, 5), 0.0)  # TODO: why < 0.0009??
    sim.v[0] = np.full((1, 5), 1.0)
    sim.b[0] = float("inf")

    result = sim.responses()
    assert isinstance(result, PNE)
    assert set(result.x[0]) == {1.0}
    assert set(result.x[1:].flatten()) == {0.0}
    assert set(result.p) == {1 / sim.q}


def test_pne_nonexistence():
    """
    Non-existence of PNE, even with 2 bidders and 2 auctions
    """
    n, m = 2, 2
    sim = Auction(n, m, no_budget=True)
    sim.v = np.array([[3 / 4, 0.0], [1.0, 1.0]])
    sim.alpha_q = np.array([1, 1])
    result = sim.responses()
    assert isinstance(result, Cycle)


def test_separate_preferences():
    """
    Each bidder has a strong preference to a separate item
    """
    n, m = 5, 5
    q = 1000
    sim = Auction(n, m, q=q, no_budget=True)
    sim.v = np.full((n, m), 0.0)
    for bidder in range(n):
        sim.v[bidder][bidder] = 1.0

    result = sim.responses()
    assert isinstance(result, PNE)
    x, p = result.x, result.p
    assert np.allclose(np.identity(n), x)
    assert np.allclose(p, 1 / q)


def test_tie():
    """
    A tie between all bidders
    """
    n, m = 5, 5
    value = 0.5
    sim = Auction(n, m, no_budget=True)
    sim.v = np.full((n, m), value)
    sim.alpha_q = np.full(n, sim.q)

    result = sim.responses()
    assert isinstance(result, PNE)
    x, p = result.x, result.p
    assert set(x.flatten()) == {1 / n}
    assert np.allclose(p, value)


def test_all_eliminated():
    """
    Every bidder is eliminated from one of the auctions
    """
    # TODO:
    pass
