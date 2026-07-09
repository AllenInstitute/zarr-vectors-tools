"""Enable ``python -m zarr_vectors_tools ...`` as an alias for the ``zvtools`` CLI."""

from zarr_vectors_tools.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
