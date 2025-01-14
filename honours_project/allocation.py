import json
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

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
