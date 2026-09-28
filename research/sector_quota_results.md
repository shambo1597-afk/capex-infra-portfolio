# Sector quotas vs the pure ranking

Design and limitations: see the docstring of research/sector_quota_study.py.
65 rebalance dates; under the live rule Capital Goods averaged 82% of the top 8.

|                                             |   mean fwd RS of top 8 (pp) |   beat Nifty 500 % |   spread of outcomes (sd, pp) |
|:--------------------------------------------|----------------------------:|-------------------:|------------------------------:|
| ('All', 'B baseline (no quotas)')           |                        8.99 |               75.4 |                         12.92 |
| ('All', 'Q one Cement + one Power')         |                        7.27 |               72.3 |                         11.71 |
| ('All', 'P balanced pillars 3/3/2')         |                        7.07 |               70.8 |                         12.09 |
| ('IS 2021-23', 'B baseline (no quotas)')    |                       14.3  |               91.4 |                         12.45 |
| ('IS 2021-23', 'Q one Cement + one Power')  |                       12.18 |               91.4 |                         10.32 |
| ('IS 2021-23', 'P balanced pillars 3/3/2')  |                       11.69 |               91.4 |                         10.41 |
| ('OOS 2024-26', 'B baseline (no quotas)')   |                        2.8  |               56.7 |                         10.63 |
| ('OOS 2024-26', 'Q one Cement + one Power') |                        1.53 |               50   |                         10.7  |
| ('OOS 2024-26', 'P balanced pillars 3/3/2') |                        1.7  |               46.7 |                         11.84 |

|                                  |   mean diff IS 2021-23 (pp) |   mean diff OOS 2024-26 (pp) |   t (full, non-overlapping) | adopt?   |
|:---------------------------------|----------------------------:|-----------------------------:|----------------------------:|:---------|
| Q one Cement + one Power minus B |                       -2.12 |                        -1.27 |                        0.39 | no       |
| P balanced pillars 3/3/2 minus B |                       -2.61 |                        -1.1  |                       -0.16 | no       |
