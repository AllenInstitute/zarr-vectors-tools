# Command-line reference

Every `zvtools` subcommand and flag, generated from the parser itself. For what
the options mean in practice, see [Convert](../convert/index.md),
[Store layout](../store_layout.md) and [Pyramids](../pyramids.md).
`python -m zarr_vectors_tools` runs the same command.

```{eval-rst}
.. argparse::
   :module: zarr_vectors_tools.cli
   :func: build_parser
   :prog: zvtools
```
