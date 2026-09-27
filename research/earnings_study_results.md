# Earnings surprise (announcement return) vs our ranking

Design and limitations: see the docstring of research/earnings_study.py.
65 rebalance dates; EAR available for 97% of eligible stock-dates. Mean rank IC of EAR with the forward 3-month RS: 0.052 (IS 0.067, OOS 0.035).

|                                                   |   mean fwd RS of top 8 (pp) |   beat Nifty 500 % |
|:--------------------------------------------------|----------------------------:|-------------------:|
| ('All', 'B baseline (rs_126_skip21)')             |                       10.01 |               80   |
| ('All', 'E1 EAR alone')                           |                        6.84 |               76.9 |
| ('All', 'E2 top-16 momentum, best 8 EAR')         |                        7.53 |               69.2 |
| ('All', 'E3 momentum, no negative EAR')           |                        8.49 |               72.3 |
| ('IS 2021-23', 'B baseline (rs_126_skip21)')      |                       15.13 |               94.3 |
| ('IS 2021-23', 'E1 EAR alone')                    |                       11.27 |               94.3 |
| ('IS 2021-23', 'E2 top-16 momentum, best 8 EAR')  |                       12.21 |               82.9 |
| ('IS 2021-23', 'E3 momentum, no negative EAR')    |                       12.29 |               88.6 |
| ('OOS 2024-26', 'B baseline (rs_126_skip21)')     |                        4.04 |               63.3 |
| ('OOS 2024-26', 'E1 EAR alone')                   |                        1.68 |               56.7 |
| ('OOS 2024-26', 'E2 top-16 momentum, best 8 EAR') |                        2.07 |               53.3 |
| ('OOS 2024-26', 'E3 momentum, no negative EAR')   |                        4.05 |               53.3 |

|                                        |   mean diff IS 2021-23 (pp) |   mean diff OOS 2024-26 (pp) |   t (full, non-overlapping) | adopt?   |
|:---------------------------------------|----------------------------:|-----------------------------:|----------------------------:|:---------|
| E1 EAR alone minus B                   |                       -3.86 |                        -2.37 |                       -0.66 | no       |
| E2 top-16 momentum, best 8 EAR minus B |                       -2.92 |                        -1.98 |                       -1.38 | no       |
| E3 momentum, no negative EAR minus B   |                       -2.84 |                         0.01 |                       -0.46 | no       |
