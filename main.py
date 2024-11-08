from simulation import Simulation


def main() -> None:
    sim = Simulation()
    allocations = sim.run()
    for a in allocations:
        print(a)


if __name__ == "__main__":
    main()
