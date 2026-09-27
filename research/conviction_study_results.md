# Conviction filter study

Design and limitations: see the docstring of research/conviction_study.py.
65 rebalance dates; the current filter changed the top 8 on 95% of them.

|                                              |   mean fwd RS of top 8 (pp) |   beat Nifty 500 % |
|:---------------------------------------------|----------------------------:|-------------------:|
| ('All', 'A current (High/Moderate)')         |                        8.08 |               76.9 |
| ('All', 'B no filter')                       |                       10.01 |               80   |
| ('All', 'C rs63 > 0')                        |                        9.64 |               76.9 |
| ('IS 2021-23', 'A current (High/Moderate)')  |                       12.48 |               94.3 |
| ('IS 2021-23', 'B no filter')                |                       15.13 |               94.3 |
| ('IS 2021-23', 'C rs63 > 0')                 |                       14.63 |               91.4 |
| ('OOS 2024-26', 'A current (High/Moderate)') |                        2.95 |               56.7 |
| ('OOS 2024-26', 'B no filter')               |                        4.04 |               63.3 |
| ('OOS 2024-26', 'C rs63 > 0')                |                        3.82 |               60   |

|                                   |   mean diff IS 2021-23 (pp) |   mean diff OOS 2024-26 (pp) |   t (full, non-overlapping) | keep?   |
|:----------------------------------|----------------------------:|-----------------------------:|----------------------------:|:--------|
| A current (High/Moderate) minus B |                       -2.65 |                        -1.09 |                       -0.78 | no      |
| C rs63 > 0 minus B                |                       -0.5  |                        -0.23 |                       -1.27 | no      |
