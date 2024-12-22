from simulation import PNE, Cycle, Simulation
import cProfile
import pstats


def main() -> None:
    sim = Simulation(10, 10, 1000)
    sim.load("data.json")

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
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
