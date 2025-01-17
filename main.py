from concurrent.futures import ProcessPoolExecutor, as_completed
import logging.handlers
import queue
import sys

from typing import TypedDict

import numpy as np
import pandas as pd
from honours_project.data import Allocation
from honours_project.simulation import PNE, Cycle, Simulation, SimulationResult
import cProfile
import pstats
import logging

# Set up logging
logger = logging.getLogger("main")
logging.basicConfig(level=logging.DEBUG)

log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(queue_handler)


def main() -> None:
    sim = Simulation(10, 5, epsilon=0.1)
    result = sim.run()
    print(result)


if __name__ == "__main__":
    listener = logging.handlers.QueueListener(log_queue)
    listener.start()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "profile":
            with cProfile.Profile() as pr:
                main()
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats()
        else:
            main()
    finally:
        listener.stop()
