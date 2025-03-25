import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

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
