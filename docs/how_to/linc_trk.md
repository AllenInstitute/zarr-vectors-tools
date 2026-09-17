# LINC TRK Ingest

The `linc_trk` ingester is used for TRK files that contain atlas label
information in `data_per_streamline["label_id"]`.

In addition to the TRK file, the ingester requires:

1. A JSON mapping file that maps TRK filenames to their expected atlas label ID.
2. A text LUT file that maps atlas label IDs to group names and RGBA colors.

The `label_id` stored in the TRK determines group membership. The LUT supplies
the metadata associated with each group.

## Input files

A LINC TRK ingest consists of three files:

```text
my_bundle.trk
linc_mapping.json
linc_lut.txt
```

The TRK must contain:

```text
data_per_streamline["label_id"]
```

There should be one `label_id` for every streamline.

Integral floating-point values such as `34.0` are accepted and converted to
integers. Non-integral, non-finite, or out-of-range values are rejected.

---

## JSON mapping file

The JSON file is a JSON object whose keys are TRK filenames and whose values
are integer atlas label IDs.

For example:

```json
{
  "my_bundle.trk": 34,
  "another_bundle.trk": 35,
  "third_bundle.trk": 42
}
```

### Rules

- The top-level JSON value must be an object.
- Each value must be an integer.
- The mapped label ID must occur in the TRK's
  `data_per_streamline["label_id"]` values.

For example, if the ingester is called with:

```text
/path/to/my_bundle.trk
```

the JSON must contain:

```json
{
  "my_bundle.trk": 34
}
```

## LUT text file

The LUT is a whitespace-delimited text file using the following format:

```text
ID NAME R G B A
```

where:

- `ID` is the integer atlas label ID.
- `NAME` is the label/group name.
- `R`, `G`, `B`, and `A` are integer color components from `0` through `255`.

For example:

```text
34 Left-Hippocampus 220 20 60 255
35 Right-Hippocampus 30 144 255 255
42 Amygdala 255 165 0 255
```

Names may contain spaces. The final four fields are always interpreted as
`R G B A`.

Comments and blank lines are allowed. Comment lines begin with `#`.

For example:

```text
# Atlas labels
34 Left-Hippocampus 220 20 60 255
35 Right-Hippocampus 30 144 255 255

# Additional structures
42 Amygdala 255 165 0 255
```

### LUT requirements

Every `label_id` that occurs in the TRK must have a corresponding entry in
the LUT.

For example, if the TRK contains:

```text
label_id = [34, 34, 35, 35]
```

the LUT must contain entries for both `34` and `35`:

```text
34 Left-Hippocampus 220 20 60 255
35 Right-Hippocampus 30 144 255 255
```

---

## How the three files work together

Suppose the TRK contains four streamlines:

```text
streamline    label_id
----------    --------
0             34
1             34
2             35
3             35
```

The mapping JSON contains:

```json
{
  "my_bundle.trk": 34
}
```

and the LUT contains:

```text
34 Left-Hippocampus 220 20 60 255
35 Right-Hippocampus 30 144 255 255
```

The resulting groups are:

```text
group 34 -> streamlines [0, 1]
group 35 -> streamlines [2, 3]
```

The group metadata comes from the LUT:

```text
group 34
    name:  Left-Hippocampus
    color: [220, 20, 60, 255]

group 35
    name:  Right-Hippocampus
    color: [30, 144, 255, 255]
```

Thus:

- `label_id` in the TRK determines **which streamlines belong to a group**.
- The LUT determines **the name and color of that group**.
- The JSON mapping identifies **the expected label for the input TRK file**.

---

## Running the ingester

The Python API is:

```python
from zarr_vectors_tools.convert.ingest.linc_trk import ingest_linc_trk

result = ingest_linc_trk(
    "my_bundle.trk",
    "my_bundle.zarr",
    chunk_shape=(64, 64, 64),
    lut_path="linc_lut.txt",
    mapping_path="linc_mapping.json",
)
```

Optional enrichment arguments work the same way as the regular TRK ingester:

```python
result = ingest_linc_trk(
    "my_bundle.trk",
    "my_bundle.zarr",
    chunk_shape=(64, 64, 64),
    lut_path="linc_lut.txt",
    mapping_path="linc_mapping.json",
    compute_length=True,
    compute_endpoints=True,
    length_range=(10.0, 500.0),
)
```

When `length_range` is supplied, streamlines outside the requested range are
removed before the groups are constructed. Group membership therefore
corresponds to the streamlines that remain in the output store.

---