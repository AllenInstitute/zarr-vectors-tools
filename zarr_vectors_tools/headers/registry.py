"""HeaderRegistry — manages format-specific headers within a store.

Headers are stored under ``/headers/<format>/.zattrs``.  The registry
provides ``add``, ``get``, ``remove``, and ``available_formats`` for
managing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Interim: replace with the public HeaderRegistry shim in zarr-vectors-py once it lands.
from zarr_vectors.core.store import FsGroup, open_store
from zarr_vectors_tools.headers.formats import Header, header_from_dict, HEADER_CLASSES


class HeaderRegistry:
    """Manages format-specific headers within a zarr vectors store.

    Args:
        store_path_or_root: Either a filesystem path (str/Path) to the
            store, or an already-open :class:`FsGroup` root handle.
    """

    def __init__(self, store_path_or_root: str | Path | FsGroup) -> None:
        if isinstance(store_path_or_root, FsGroup):
            self._root = store_path_or_root
        else:
            self._root = open_store(str(store_path_or_root), mode="r+")

    def _headers_group(self, create: bool = False) -> FsGroup:
        """Get or create the /headers/ group."""
        if create:
            return self._root.require_group("headers")
        if "headers" not in self._root:
            raise KeyError("No /headers/ group in store")
        return self._root["headers"]

    @property
    def available_formats(self) -> list[str]:
        """List of format names with stored headers."""
        try:
            hg = self._headers_group()
        except KeyError:
            return []
        return sorted(
            name for name in hg
            if not name.startswith(".")
        )

    def has(self, format_name: str) -> bool:
        """Check if a header exists for the given format."""
        return format_name in self.available_formats

    def get(self, format_name: str) -> Header:
        """Read and deserialise a format header.

        Args:
            format_name: Format identifier (e.g. ``"trk"``, ``"swc"``).

        Returns:
            The deserialised :class:`Header` subclass.

        Raises:
            KeyError: If no header exists for this format.
        """
        try:
            hg = self._headers_group()
        except KeyError:
            raise KeyError(f"No header stored for format '{format_name}'")

        if format_name not in hg:
            raise KeyError(f"No header stored for format '{format_name}'")

        fmt_group = hg[format_name]
        attrs = fmt_group.attrs.to_dict()
        return header_from_dict(attrs)

    def add(self, format_name: str, header: Header) -> None:
        """Store a format header.

        If a header for this format already exists, it is overwritten.

        Args:
            format_name: Format identifier.
            header: Header dataclass to store.
        """
        hg = self._headers_group(create=True)
        fmt_group = hg.require_group(format_name)
        fmt_group.attrs.update(header.to_dict())

    def remove(self, format_name: str) -> None:
        """Remove a stored header.

        Args:
            format_name: Format to remove.

        Raises:
            KeyError: If no header exists for this format.
        """
        try:
            hg = self._headers_group()
        except KeyError:
            raise KeyError(f"No header stored for format '{format_name}'")

        if format_name not in hg:
            raise KeyError(f"No header stored for format '{format_name}'")

        import shutil
        fmt_path = hg.path / format_name
        shutil.rmtree(fmt_path)

    def __repr__(self) -> str:
        fmts = self.available_formats
        return f"HeaderRegistry(formats={fmts})"
