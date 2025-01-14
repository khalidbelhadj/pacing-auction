import numpy as np
from numpy.typing import NDArray


def subsequent(bidder: int, auction: int, mask: NDArray[np.bool_]) -> None:
    mask[bidder, auction:] = False


def all(bidder: int, auction: int, mask: NDArray[np.bool_]) -> None:
    mask[bidder, :] = False


def current(bidder: int, auction: int, mask: NDArray[np.bool_]) -> None:
    mask[bidder, auction] = False
