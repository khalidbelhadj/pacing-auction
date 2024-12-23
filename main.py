import sys
from simulation import PNE, Cycle, Simulation
import cProfile
import pstats
import matplotlib.pyplot as plt


def test_simulation(sim: Simulation, runs: int) -> tuple[list[PNE], list[Cycle]]:
    pnes: list[PNE] = []
    cycles: list[Cycle] = []

    for i in range(runs):
        sim.reset()
        print(100 * i / runs, "%")
        result = sim.run()
        match result:
            case PNE():
                pnes.append(result)
            case Cycle():
                cycles.append(result)
            case _:
                pass
    return pnes, cycles


def main() -> None:
    sim = Simulation(10, 4, 1000)
    # match sim.run():
    #     case PNE(t, a, i):
    #         print(f"PNE: {i}")
    #         for i in a:
    #             print(i)
    #     case Cycle(a, i):
    #         print(f"Cycle: {i}")
    #     case _:
    #         pass
    # return
    pnes, cycles = test_simulation(sim, 10)

    print(f"PNEs: {len(pnes)}")
    print(f"Cycles: {len(cycles)}")
    print()
    print(f"PNE mean iters: {sum(pne.iteration for pne in pnes) / len(pnes)}")
    print(f"Cycle mean iters: {sum(cycle.iteration for cycle in cycles) / len(cycles)}")
    print()
    print(f"PNE mean time: {sum(pne.time for pne in pnes) / len(pnes)}")
    print(f"Cycle mean time: {sum(cycle.time for cycle in cycles) / len(cycles)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        with cProfile.Profile() as pr:
            main()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()
    else:
        main()
