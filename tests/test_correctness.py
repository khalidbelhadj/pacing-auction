import numpy as np
from honours_project.data import PNE, Cycle
from honours_project.simulation import Simulation
import logging

logger = logging.getLogger("test_correctness")


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

    sim.alpha_q[0] = 1
    sim.alpha_q[1] = 1

    result = sim.run()
    assert isinstance(result, Cycle)


def test_seperate_preferences():
    """
    Each bidder has a strong preference to a seperate item
    """
    sim = Simulation(5, 5, no_budget=True)
    sim.v = np.full((5, 5), 0.0)
    for bidder in range(5):
        sim.v[bidder][bidder] = 1.0

    result = sim.run()
    assert isinstance(result, PNE)
    assert all(
        a.bidder == i and a.auction == i and a.price == 1 / sim.q
        for i, a in enumerate(result.allocations)
    )


def test_same_after_rerun():
    """
    The same result is returned after re-running the simulation
    """
    pne_count = 0
    cycle_count = 0

    while pne_count < 5 or cycle_count < 5:
        sim = Simulation(5, 5)
        result1 = sim.run()
        result2 = sim.run()

        if type(result1) != type(result2):
            logger.error(f"{sim.seed=}")
        assert type(result1) == type(result2)
        # assert result2.iteration == 0 # TODO: why

        if isinstance(result1, PNE) and isinstance(result2, PNE):
            pne_count += 1
            assert result1.allocations == result2.allocations
        elif isinstance(result1, Cycle) and isinstance(result2, Cycle):
            cycle_count += 1
