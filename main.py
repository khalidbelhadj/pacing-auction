import json
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

# Constants
TYPE_PNE = 0
TYPE_CYCLE = 1
SIM_COUNT = 1
q = 1000
n = 10
m = 10


class Entry(TypedDict):
    id: int
    n: int
    m: int
    q: int
    time: float
    iteration: int
    type: int
    allocations: list


def sample(base_id: int, n: int, m: int, q: int) -> list[Entry]:
    logger.info(f"Started: n={n}, m={m}, q={q}, count={SIM_COUNT}")
    entries = []

    for i in range(SIM_COUNT):
        sim = Simulation(n, m, q)
        id = base_id + i

        result = sim.run()
        time = result.time
        iteration = result.iteration
        allocations = result.allocations if isinstance(result, PNE) else []

        entries.append(
            Entry(
                {
                    "id": id,
                    "n": n,
                    "m": m,
                    "q": q,
                    "time": time,
                    "iteration": iteration,
                    "type": TYPE_PNE if isinstance(result, PNE) else TYPE_CYCLE,
                    "allocations": [a.__dict__ for a in allocations],
                }
            )
        )

    return entries


def main() -> None:
    global n, m

    result: list[Entry] = []
    for n in range(1, n + 1):
        for m in range(1, m + 1):
            base_id = n * (n - 1) + m - 1
            base_id *= SIM_COUNT
            result.extend(sample(base_id, n, m, q))

        with open("data/output.json", "w") as f:
            json.dump(result, f, indent=2)


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
