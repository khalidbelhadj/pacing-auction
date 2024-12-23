from typing import Protocol


class Elimination(Protocol):
    """
    An interface for elimination procedures
    """

    def __init__(self, n: int, m: int) -> None:
        """
        Initialize
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

    def clear(self) -> None:
        """
        Clear all eliminations
        """
        ...


class All:
    """
    Eliminate the bidder from all auctions
    """

    def __init__(self, n: int, m: int) -> None:
        self.eliminated: set[int] = set()

    def eliminate(self, bidder: int, auction: int) -> None:
        self.eliminated.add(bidder)

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        return bidder in self.eliminated

    def clear(self) -> None:
        self.eliminated.clear()


class Subsequent:
    """
    Eliminate the bidder from the auction and all subsequent auctions
    """

    def __init__(self, n: int, m: int) -> None:
        self.eliminated: list[int] = [-1] * n

    def eliminate(self, bidder: int, auction: int) -> None:
        self.eliminated[bidder] = auction

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        return self.eliminated[bidder] >= auction

    def clear(self) -> None:
        self.eliminated = [-1] * len(self.eliminated)


class Current:
    """
    Eliminate the bidder from a specific auction
    """

    def __init__(self, n: int, m: int) -> None:
        self.eliminated: list[list[bool]] = [[False] * m for _ in range(n)]

    def eliminate(self, bidder: int, auction: int) -> None:
        self.eliminated[bidder][auction] = True

    def is_eliminated(self, bidder: int, auction: int) -> bool:
        return self.eliminated[bidder][auction]

    def clear(self) -> None:
        self.eliminated.clear()
