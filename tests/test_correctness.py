import numpy as np
from honours_project.data import PNE, Cycle
from honours_project.simulation import Simulation


def test_dominant_player():
    """
    Single dominant player with the highest budget and valuation for all auctions
    """
    sim = Simulation(5, 5)
    sim.v = np.full((5, 5), 0.049)  # TODO: why < 0.05??
    sim.v[0] = np.full((1, 5), 1.0)
    sim.b[0] = float("inf")

    result = sim.run()
    assert isinstance(result, PNE)
    assert all(a.bidder == 0 for a in result.allocations)


def test_pne_nonexistence():
    """
    Non-existence of PNE, even with 2 bidders and 2 auctions
    """
    sim = Simulation(2, 2, no_budget=True)
    sim.v[0] = np.array([1.0, 1.0])
    sim.v[1] = np.array([3 / 4, 0])

    sim.alpha[0] = 1
    sim.alpha[1] = 1

    result = sim.run()
    assert isinstance(result, Cycle)
