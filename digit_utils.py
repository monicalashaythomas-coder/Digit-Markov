"""
Digit extraction and rolling buffers.

IMPORTANT: last-digit extraction from a price MUST be done on the fixed-decimal
string representation, not on a float cast, or trailing zeros silently vanish
(e.g. 123.40 -> "123.4" -> last digit "4" is WRONG if the contract's pip size
implies 2 decimals; the true last digit is 0). This mirrors the bug caught in
digit_ev_validator.py. Deriv ticks carry a `pip_size` (decimal places) per
symbol from the `ticks_history`/`tick` response — always format to that many
decimals before reading the final character.
"""
from collections import deque
from typing import Deque, Optional


def extract_last_digit(price: float, pip_size: int) -> int:
    """
    Extract the last significant digit of `price` at `pip_size` decimal places.

    pip_size is the number of decimal places Deriv quotes for this symbol
    (obtained from the API, e.g. active_symbols / tick response). Formatting
    to a fixed width avoids float-repr trailing-zero loss.
    """
    if pip_size < 0:
        raise ValueError("pip_size must be >= 0")
    formatted = f"{price:.{pip_size}f}"
    digit_char = formatted[-1]
    if not digit_char.isdigit():
        raise ValueError(f"Could not extract digit from formatted price {formatted!r}")
    return int(digit_char)


class DigitBuffer:
    """Fixed-size rolling buffer of last digits for one symbol."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._buf: Deque[int] = deque(maxlen=maxlen)

    def push(self, digit: int) -> None:
        if not (0 <= digit <= 9):
            raise ValueError(f"digit out of range: {digit}")
        self._buf.append(digit)

    def __len__(self) -> int:
        return len(self._buf)

    def as_list(self):
        return list(self._buf)

    def is_warm(self, min_len: int) -> bool:
        return len(self._buf) >= min_len

    def last(self) -> Optional[int]:
        return self._buf[-1] if self._buf else None
