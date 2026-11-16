# GPU read benchmark -- supplementary table

`read_cells` over every cell of a point cloud, vertices + vertex_attributes/intensity; mean of 5 runs after one warm-up, ±95 % CI (Student's t). Measured on NVIDIA RTX A2000 12GB, x86_64; zarr_vectors 0.9.3.dev39+gcd6a3c0e4.d20260924 (06a3bc8), cupy 13.6.0, nvCOMP 4.2.0.

| codec | n_vertices | cpu | cpu ±95% | gpu (host decode) | gpu (host decode) ±95% | gpu (device decode) | gpu (device decode) ±95% | speed-up, GPU, device decode | speed-up, GPU, host decode |
|---|---|---|---|---|---|---|---|---|---|
| none | 1,000 | 9.91 ms | 1.05 ms | 10.24 ms | 0.57 ms | 8.08 ms | 0.65 ms | 1.23× | 0.97× |
| none | 3,000 | 12.46 ms | 0.83 ms | 12.75 ms | 0.28 ms | 10.47 ms | 0.22 ms | 1.19× | 0.98× |
| none | 10,000 | 15.68 ms | 0.77 ms | 13.38 ms | 0.49 ms | 10.63 ms | 1.08 ms | 1.48× | 1.17× |
| none | 30,000 | 17.71 ms | 0.85 ms | 15.02 ms | 2.42 ms | 10.90 ms | 0.26 ms | 1.63× | 1.18× |
| none | 100,000 | 12.95 ms | 2.31 ms | 12.81 ms | 0.64 ms | 10.83 ms | 0.21 ms | 1.20× | 1.01× |
| none | 300,000 | 15.28 ms | 2.24 ms | 13.19 ms | 1.79 ms | 9.07 ms | 0.69 ms | 1.68× | 1.16× |
| none | 1,000,000 | 15.30 ms | 1.97 ms | 16.58 ms | 2.04 ms | 12.61 ms | 0.90 ms | 1.21× | 0.92× |
| none | 3,000,000 | 54.62 ms | 4.33 ms | 60.29 ms | 0.61 ms | 21.71 ms | 1.09 ms | 2.52× | 0.91× |
| none | 10,000,000 | 181.61 ms | 11.80 ms | 193.80 ms | 2.17 ms | 79.38 ms | 1.73 ms | 2.29× | 0.94× |
| zstd | 1,000 | 8.60 ms | 0.28 ms | 8.58 ms | 0.06 ms | 12.26 ms | 2.84 ms | 0.70× | 1.00× |
| zstd | 3,000 | 9.00 ms | 0.31 ms | 9.94 ms | 1.39 ms | 10.34 ms | 0.48 ms | 0.87× | 0.91× |
| zstd | 10,000 | 11.10 ms | 0.28 ms | 11.15 ms | 1.44 ms | 11.39 ms | 0.66 ms | 0.97× | 1.00× |
| zstd | 30,000 | 11.05 ms | 0.87 ms | 11.38 ms | 0.97 ms | 10.98 ms | 1.34 ms | 1.01× | 0.97× |
| zstd | 100,000 | 10.55 ms | 1.12 ms | 10.58 ms | 0.99 ms | 10.60 ms | 1.30 ms | 0.99× | 1.00× |
| zstd | 300,000 | 14.97 ms | 1.05 ms | 15.05 ms | 1.28 ms | 14.52 ms | 0.74 ms | 1.03× | 0.99× |
| zstd | 1,000,000 | 29.03 ms | 1.19 ms | 29.67 ms | 1.44 ms | 25.82 ms | 0.86 ms | 1.12× | 0.98× |
| zstd | 3,000,000 | 63.47 ms | 6.95 ms | 65.19 ms | 2.18 ms | 39.32 ms | 2.61 ms | 1.61× | 0.97× |
| zstd | 10,000,000 | 252.16 ms | 9.80 ms | 268.16 ms | 5.33 ms | 84.42 ms | 2.63 ms | 2.99× | 0.94× |
