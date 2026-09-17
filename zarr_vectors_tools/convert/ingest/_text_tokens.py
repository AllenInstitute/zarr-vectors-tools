"""Vectorised tokenising for the whitespace-separated ASCII mesh formats.

The OBJ and ASCII STL readers walk a file one line at a time, calling
``str.strip`` and ``str.split`` on every line and ``float`` or ``int`` on
every field.  That is correct, and it takes hours on a 100M-face mesh.  This
module finds the same tokens with numpy over the raw bytes, so a reader only
drops to Python for the one conversion numpy cannot reproduce exactly:
``float`` itself, which is correctly rounded where numpy's and pandas' fast
text parsers are not.

Everything here is a fast path in front of the line-by-line reader, not a
replacement for it.  :func:`read_plain_ascii` only accepts text whose lines
and tokens are provably the ones ``open(path)`` iteration and ``str.split()``
would produce, and the readers return ``None`` for anything else they cannot
parse exactly.  The caller then runs its original loop, so unusual input
keeps its old result and a malformed file keeps its old error message.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Files are tokenised in pieces of about this many bytes, cut at line ends,
# so the per-byte masks stay small however large the mesh is.  Read at call
# time so tests can force many chunks through a small file.
_CHUNK_BYTES = 1 << 23

# ``str.split()`` and ``str.strip()`` also treat vertical tab, form feed and
# the four ASCII separator characters as whitespace.  They almost never
# appear in a mesh file, so rather than teach the byte tokenizer about them,
# a file containing any of them takes the line-by-line reader.
_RARE_WHITESPACE = re.compile(rb"[\x0b\x0c\x1c-\x1f]")

_SPACE, _TAB, _NEWLINE = 0x20, 0x09, 0x0A


class TokenTable(NamedTuple):
    """Token boundaries for one chunk of text.

    Token ``k`` of line ``i`` is ``arr[starts[first[i] + k]:ends[first[i] + k]]``
    for ``k < count[i]``.
    """

    arr: np.ndarray  # (N,) uint8, always ending in a newline
    starts: np.ndarray  # (T,) first byte of each token
    ends: np.ndarray  # (T,) one past the last byte of each token
    first: np.ndarray  # (L,) index of each line's first token
    count: np.ndarray  # (L,) number of tokens on each line


def read_plain_ascii(path: Path) -> bytes | None:
    """Read *path* for byte-level tokenising, or return ``None``.

    Returns the file's bytes with universal newlines applied, exactly as text
    mode would present them, when the bytes decode to the same characters
    under the encoding ``open(path)`` would use and contain no whitespace
    beyond space, tab and line breaks.  Returns ``None`` otherwise, which
    tells the caller to use its line-by-line reader.
    """
    # Opened in text mode only to learn the encoding the line-by-line reader
    # decodes with; the bytes come from the buffer underneath, unread so far.
    with open(path) as f:
        encoding = f.encoding
        data = f.buffer.read()
    if not data.isascii() or not _ascii_decodes_to_itself(encoding):
        return None
    if _RARE_WHITESPACE.search(data) is not None:
        return None
    if b"\r" in data:
        # Text mode turns both "\r\n" and a lone "\r" into "\n".
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


@functools.cache
def _ascii_decodes_to_itself(encoding: str) -> bool:
    ascii_bytes = bytes(range(128))
    try:
        return ascii_bytes.decode(encoding) == ascii_bytes.decode("ascii")
    except (LookupError, UnicodeDecodeError):
        return False


def iter_chunks(data: bytes) -> Iterator[np.ndarray]:
    """Yield *data* as uint8 arrays of whole lines, each ending in a newline."""
    view = memoryview(data)
    size = len(data)
    pos = 0
    while pos < size:
        if size - pos <= _CHUNK_BYTES:
            cut = size
        else:
            cut = data.rfind(b"\n", pos, pos + _CHUNK_BYTES) + 1
            if cut == 0:
                # One line longer than a whole chunk: take all of it.
                cut = data.find(b"\n", pos + _CHUNK_BYTES) + 1 or size
        if cut == size and data[-1] != _NEWLINE:
            # A final line without a newline reads the same as one with it;
            # adding it lets every chunk assume lines end in a newline.
            yield np.frombuffer(bytes(view[pos:cut]) + b"\n", dtype=np.uint8)
        else:
            yield np.frombuffer(view[pos:cut], dtype=np.uint8)
        pos = cut


def tokenize(arr: np.ndarray) -> TokenTable:
    """Split a chunk into whitespace-separated tokens, as ``str.split()`` would."""
    blank = (arr == _SPACE) | (arr == _TAB) | (arr == _NEWLINE)
    filled = ~blank
    head = filled.copy()
    head[1:] &= blank[:-1]
    tail = filled
    tail[:-1] &= blank[1:]
    starts = np.flatnonzero(head)
    ends = np.flatnonzero(tail) + 1
    newlines = np.flatnonzero(arr == _NEWLINE)
    tokens_before = np.searchsorted(starts, newlines)
    first = np.zeros_like(tokens_before)
    first[1:] = tokens_before[:-1]
    return TokenTable(arr, starts, ends, first, tokens_before - first)


def line_heads(table: TokenTable) -> np.ndarray:
    """Byte offset of each line's first token.

    Meaningless for lines with no tokens, which callers must mask out with
    ``table.count > 0``.
    """
    if table.starts.size == 0:
        return np.zeros(table.first.size, dtype=np.int64)
    return table.starts[np.minimum(table.first, table.starts.size - 1)]


def starts_with(arr: np.ndarray, offsets: np.ndarray, word: bytes) -> np.ndarray:
    """Whether the bytes at each offset begin with *word*.

    A chunk always ends in a newline and *word* never contains one, so a
    window that runs off the end of the chunk simply fails to match.
    """
    match = np.ones(offsets.size, dtype=bool)
    last = arr.size - 1
    for k, byte in enumerate(word):
        match &= arr[np.minimum(offsets + k, last)] == byte
    return match


def parse_floats(
    table: TokenTable, lines: np.ndarray, offset: int, width: int = 3,
) -> np.ndarray | None:
    """Parse tokens ``offset .. offset + width - 1`` of each line with ``float``.

    Every selected line must have at least ``offset + width`` tokens.  Returns
    a ``(len(lines), width)`` float64 array, or ``None`` when any field is not
    something ``float`` accepts, so the caller can reproduce the error.
    """
    if lines.size == 0:
        return np.empty((0, width), dtype=np.float64)
    arr = table.arr
    first = table.first[lines] + offset
    lo = table.starts[first]
    hi = table.ends[first + width - 1]
    # Blank out every byte outside the wanted spans, then let bytes.split()
    # hand over the fields.  A span starts on a token byte and ends on a
    # blank one, so no start can coincide with an end.
    edge = np.zeros(arr.size, dtype=np.int8)
    edge[lo] = 1
    edge[hi] = -1
    keep = np.cumsum(edge, dtype=np.int8).view(np.bool_)
    fields = np.where(keep, arr, np.uint8(_SPACE)).tobytes().split()
    if len(fields) != width * lines.size:
        return None
    try:
        values = np.fromiter(map(float, fields), dtype=np.float64, count=len(fields))
    except ValueError:
        return None
    return values.reshape(-1, width)
