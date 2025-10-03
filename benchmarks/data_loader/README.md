# Data Loading Benchmarks

Comparison of MultiViewTransform vs RoundRobin data loading approaches for multi-view SSL.

## Results

| Batch Size | Approach | Load Time (ms) | Transfer Time (ms) | Throughput (img/s) |
|------------|----------|----------------|-------------------|-------------------|
| 256 | MultiViewTransform | 789.61 ± 1852.69 | 6.51 ± 2.59 | **321.6** |
| 256 | RoundRobin | 857.45 ± 2024.98 | 8.20 ± 3.94 | 295.7 |
| 1024 | MultiViewTransform | 3129.27 ± 6066.61 | 24.83 ± 4.39 | **324.7** |
| 1024 | RoundRobin | 3388.74 ± 7529.84 | 22.65 ± 0.84 | 300.2 |
