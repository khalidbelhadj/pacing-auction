# Pacing Auction Simulator

A Python framework for simulating and analysing first-price pacing auctions with budget constraints.

## Project Overview

This project implements a comprehensive simulator for first-price auctions where bidders have budget constraints and use pacing multipliers to scale down their bids. The simulator can find pure Nash equilibria (PNE) or cycles in bidder strategies.

Key features:
- Multiple auction generators (complete, sampled, correlated)
- Various elimination strategies for bidders who violate budget constraints
- Multithreaded best response dynamics for improved performance
- Comprehensive metrics and statistics collection
- Visualization tools for analysing results

## Project Structure

```
pacing_auction/              # Main package directory
├── auction.py               # Core auction implementation
├── data.py                  # Data classes and structures
├── elimination.py           # Elimination strategies for budget violations
├── generator.py             # Auction state generators
└── __init__.py              # Package initialization

scripts/                     # Utility scripts
├── results.py               # Script for running comprehensive test suites
└── example.py               # Example script for running a simple auction

notebooks/                   # Jupyter notebooks for analysis and visualization

tests/                       # Unit and integration tests
```

## Requirements
The project requires Python 3.13+ due to the [optional GIL](https://peps.python.org/pep-0703/). The python interpreter must be built with the `--disable-gil` flag set. Additionally, the environment should have the variable `PYTHON_GIL=0` to disable the GIL.

While the simulation still works with the GIL enabled, the performance will be significantly worse due to the lack of parallelism.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khalidbelhadj/pacing-auction.git
   cd pacing-auction
   ```

2. Create and activate a virtual environment (Python 3.13+ required):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   or using the [uv project manager](https://github.com/astral-sh/uv)
   ```bash
   uv sync
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage Example

Below is a simple example of creating and running an auction simulation:

```python
from pacing_auction.auction import Auction
from pacing_auction.data import PNE, Cycle
from pacing_auction.generator import CompleteAuctionGenerator

# Create an auction with 3 bidders and 4 items
auction = Auction(
    n=3,  # Number of bidders
    m=4,  # Number of items
    q=1000,  # Granularity of pacing multipliers
    seed=42,  # Random seed for reproducibility
    generator=CompleteAuctionGenerator(),  # Valuation generator
)

# Run best response dynamics to find PNE or cycle
result = auction.responses()

match result:
    case PNE(iteration, x, p, stats=stats):
        print(f"Found PNE in {result.iteration} iterations")

        # Calculate metrics
        social_welfare = auction.social_welfare(x, p)
        liquid_welfare = auction.liquid_welfare(x, p)
        revenue = auction.revenue(x, p)

        print(f"Social welfare: {social_welfare}")
        print(f"Liquid welfare: {liquid_welfare}")
        print(f"Revenue: {revenue}")
    case Cycle(iteration, stats=stats):
        print(f"Found cycle after {result.iteration} iterations")
        print(f"Cycle length: {result.stats['cycle_length']}")
    case _:
        pass
```

## Running Experiments

The project includes a comprehensive test suite to evaluate auction behavior across different parameters:

```bash
python scripts/results.py --min-n 2 --max-n 5 --min-m 2 --max-m 5 --runs 10
```
use `--help` to look at other parameters

## Understanding the Code

### Auction Class

The core `Auction` class in `auction.py` implements the first-price pacing auction mechanism. It handles:

- Bidder valuations and budget constraints
- Pacing multipliers (alpha) for each bidder
- Best response dynamics to find equilibria
- Metrics calculation (social welfare, liquid welfare, revenue)

### Auction Generators

Different generators create various valuation matrices:

- `CompleteAuctionGenerator`: Generates valuations with full competition
- `SampledAuctionGenerator`: Generates sparse valuation matrices
- `CorrelatedAuctionGenerator`: Generates valuations with controllable correlation

### Elimination Strategies

When bidders violate budget constraints, different elimination strategies can be applied:
- `Subsequent`: Eliminate the bidder from the current and all subsequent auctions
- `Current`: Eliminate the bidder only from the current auction
- `All`: Eliminate the bidder from all auctions

## License

[MIT License](LICENSE)
