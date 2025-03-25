from concurrent.futures import ProcessPoolExecutor, as_completed
import logging.handlers
import threading
import time

from matplotlib import pyplot as plt
import pandas as pd
import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from pacing_auction.data import BRDResult
from pacing_auction.auction import Auction
import logging

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)


def print_result(sim: Auction, res: BRDResult):
    print(f"Seed:               {sim.seed}")
    print(f"Result:             {type(res)}, {res.iteration}")
    print(f"Time:               {round(res.stats['time'], 2)}s")
    print(f"Time per Iteration: {1000 * round(res.stats["time"] / res.iteration, 4)}ms")


def run_sim(n, m, number_of_simulations):
    result = []
    for _ in range(number_of_simulations):
        sim = Auction(n, m, collect_stats=True)
        res = sim.responses()
        print_result(sim, res)
        result.append((sim.n, sim.m, res))
    return result


def main() -> None:
    logger.info("Starting")
    sim = Auction(5, 5, shuffle=True)
    res = sim.responses()

    # social_welfare = res.stats["social_welfare"]
    # liquid_welfare = res.stats["liquid_welfare"]

    # plt.plot(social_welfare)
    # plt.plot(liquid_welfare)
    # plt.show()

    print_result(sim, res)


if __name__ == "__main__":
    main()
