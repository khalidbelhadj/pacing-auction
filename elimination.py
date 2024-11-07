import collections
from typing import Protocol


class Elimination(Protocol):
    """
    An interface for elimination procedures
    """

    def __init__(self) -> None:
        """
        Initialize the elimination procedure. Some procedures require extra state
        """
        ...

    def eliminate(self, bidder: int, auction: int) -> None:
        """
        Eliminate the bidder, given the auction
        """
        ...

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        """
        Check if the bidder is eliminated, given the auction
        """
        ...


class Subsequent:
    """
    Eliminate the bidder from the auction and all subsequent auctions
    """

    def __init__(self) -> None:
        self.eliminated: dict[int, int] = dict()

    def eliminate(self, bidder: int, auction: int) -> None:
        self.eliminated[bidder] = auction

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        return self.eliminated.get(bidder, -1) >= auction


class All:
    """
    Eliminate the bidder from all auctions
    """

    def __init__(self) -> None:
        self.eliminated: dict[int, set[int]] = collections.defaultdict(set)

    def eliminate(self, bidder: int, auction: int) -> None:
        self.eliminated[bidder].add(auction)

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        return auction in self.eliminated[bidder]
