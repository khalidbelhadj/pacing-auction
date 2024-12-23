import sys
from simulation import PNE, Cycle, Simulation
import cProfile
import pstats


def main() -> None:
    sim = Simulation(15, 15, 1000)
    sim.save("data.json")
    # sim.load("data.json")

    match sim.run():
        case Cycle(iteration):
            print("Cycle:", iteration)
            return
        case PNE(allocations, iteration):
            print("Allocations:")
            allocations.sort(key=lambda a: a.auction)
            for a in allocations:
                print("    - ", a)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        with cProfile.Profile() as pr:
            main()
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()
    else:
        main()
