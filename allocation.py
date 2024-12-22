from typing import NamedTuple


class Allocation(NamedTuple):
    """
    An allocation of the auction, signifying the winner
    """

    bidder: int
    auction: int
    price: float

    def __repr__(self) -> str:
        return f"Allocation(bidder={self.bidder}, auction={self.auction}, price={self.price})"
