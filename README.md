# zarr-vectors-tools

> [!NOTE]
> This package is under development and will change.

**Ingest and export workflows for the Zarr Vector Format (ZVF).**

`zarr-vectors-tools` is the companion workflow package to [`zarr-vectors-py`](https://github.com/Andrew-Keenlyside/zarr-vectors-py). The core read/write APIs (chunking, sharding, multiresolution, spatial binning) live in the `zarr_vectors` package. This package adds format-conversion workflows that wrap third-party readers and writers, plus example notebooks demonstrating end-to-end pipelines.

*Aligned to the Zarr Vectors specification by Forrest Collman, Allen Institute for Brain Sciences.*
[Specification](https://github.com/AllenInstitute/zarr_vectors)

---

## Install

```bash
pip install zarr-vectors-tools
```

## Supported formats

| Format    | Ingest | Export | Geometry        | Backend    |
|-----------|:------:|:------:|-----------------|------------|
| CSV       | yes    | yes    | points          | numpy      |
| LAS/LAZ   | yes    |        | points          | laspy      |
| PLY       | yes    | yes    | points / meshes | plyfile    |
| OBJ       | yes    | yes    | meshes          | pure-python |
| STL       | yes    |        | meshes          | pure-python |
| SWC       | yes    | yes    | skeletons       | numpy      |
| GraphML   | yes    |        | graphs          | networkx   |
| TRK       | yes    | yes    | streamlines     | nibabel    |
| TCK       | yes    |        | streamlines     | nibabel    |
| TRX       | yes    | yes    | streamlines     | trx-python |
