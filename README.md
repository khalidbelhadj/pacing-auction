# Pacing Auction Simulation

2024-2025 honours project

## Project Structure

### Module Descriptions

- **auction.py**: Contains the main `Auction` class that implements:
  - First Price Auction (FPA) mechanism
  - Utility calculations with caching
  - Best response computations (single-threaded and multi-threaded)
  - PNE finding and cycle detection
  - NumPy-optimized bid calculations

- **data.py**: Defines core data structures:
  - `Distribution` classes for value/alpha sampling
  - Result types (`PNE`, `Cycle`, `Violation`)
  - Allocation and Best Response data structures

- **elimination.py**: Implements strategies for handling budget violations:
  - `ElimStrategy` base class
  - Implementations of specific elimination strategies

## Getting Started

### Option 1: Traditional Python Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd honours-project

# Create and activate a virtual environment
python -m venv venv

# On Unix/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Clone the repository
git clone <repository-url>
cd honours-project

# Run directly using uv
uv run main.py
```

## Usage

```python
from honours_project.auction import Auction

# Create an auction with 3 bidders and 2 items
auction = Auction(
    n=3,  # number of bidders
    m=2,  # number of items
)

# Run the simulation
result = auction.run()
```

- **Core Parameters**:
  - `n`: Number of bidders
  - `m`: Number of items
  - `q`: Discretization parameter (default: 1000)
  - `epsilon`: Minimum utility improvement threshold

- **Performance Flags**:
  - `threaded`: Enable multi-threaded best response computation
  - `cache_utility`: Enable utility calculation caching
  - `collect_stats`: Enable collection of simulation statistics

- **Sampling Control**:
  - `seed`: Random seed for reproducibility
  - `v_dist`: Custom distribution for item valuations
  - `alpha_q_dist`: Custom distribution for alpha parameters

## Advanced Features

### Custom Value Distributions

```python
from honours_project.data import Uniform

# Create auction with custom value distribution
auction = Auction(
    n=3,
    m=2,
    v_dist=Uniform(0, 10)  # Values between 0 and 10
)
```

### Performance Monitoring

```python
# Run with statistics collection
auction = Auction(n=3, m=2, collect_stats=True)
result = auction.run()

# Access statistics
print(f"Cache hits: {auction.utility_cache_hits}")
print(f"Cache misses: {auction.utility_cache_misses}")
print(f"Time taken: {result.stats['time']:.2f}s")
```

### Save/Load Functionality

```python
# Save auction state
auction.save("auction_state.json")

# Load auction state
loaded_auction = Auction.load("auction_state.json")
