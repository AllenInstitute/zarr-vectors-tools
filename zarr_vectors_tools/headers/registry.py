"""HeaderRegistry — typed format headers on top of core's dict registry.

Headers live under ``/headers/<format>/.zattrs``.  Core owns the storage
half (:class:`zarr_vectors.headers.HeaderRegistry`), which round-trips
opaque JSON dicts and knows nothing about formats; this subclass adds the
half that belongs to a format package — turning those dicts into the
:class:`~zarr_vectors_tools.headers.formats.Header` dataclasses, and back.

This used to be a full copy of core's class.  The copy carried two bugs
that core has since fixed, and that a copy could only ever fix twice:
``isinstance(..., FsGroup)`` (``FsGroup`` is returned only when the
backing store is a ``LocalStore``, so a cloud-backed root fell through to
``open_store(str(root))`` and tried to open a stringified Group as a path)
and ``shutil.rmtree(hg.path)`` in ``remove`` (``.path`` raises for any
non-local store, so removing a header worked on disk and nowhere else).
Subclassing is what stops that from happening a third time.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.headers import HeaderRegistry as _CoreHeaderRegistry

from zarr_vectors_tools.headers.formats import Header, header_from_dict


class HeaderRegistry(_CoreHeaderRegistry):
    """Manages format-specific headers within a zarr vectors store.

    Same constructor as core's — a path/URL, or an already-open ``Group``
    root handle.  ``add`` and ``get`` speak :class:`Header` rather than
    ``dict``; ``available_formats``, ``has`` and ``remove`` are inherited
    unchanged.
    """

    def get(self, format_name: str) -> Header:  # type: ignore[override]
        """Read and deserialise a format header.

        Args:
            format_name: Format identifier (e.g. ``"trk"``, ``"swc"``).

        Returns:
            The deserialised :class:`Header` subclass.

        Raises:
            KeyError: If no header exists for this format.
        """
        return header_from_dict(super().get(format_name))

    def add(self, format_name: str, header: Header) -> None:  # type: ignore[override]
        """Store a format header.

        If a header for this format already exists, it is overwritten.

        Args:
            format_name: Format identifier.
            header: Header dataclass to store.
        """
        payload: dict[str, Any] = (
            header.to_dict() if isinstance(header, Header) else dict(header)
        )
        super().add(format_name, payload)

    def __repr__(self) -> str:
        return f"HeaderRegistry(formats={self.available_formats})"
