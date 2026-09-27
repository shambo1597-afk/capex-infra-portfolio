# Momentum refinements: residual momentum and frog-in-the-pan

Design and limitations: see the docstring of research/signal_extensions_study.py.
64 rebalance dates.

|                                               |   mean fwd RS of top 8 (pp) |   beat Nifty 500 % |
|:----------------------------------------------|----------------------------:|-------------------:|
| ('All', 'B baseline (rs_126_skip21)')         |                        9.74 |               79.7 |
| ('All', 'R residual momentum')                |                        7.5  |               75   |
| ('All', 'F frog-in-the-pan')                  |                        5.42 |               62.5 |
| ('IS 2021-23', 'B baseline (rs_126_skip21)')  |                       14.77 |               94.1 |
| ('IS 2021-23', 'R residual momentum')         |                       12.31 |               91.2 |
| ('IS 2021-23', 'F frog-in-the-pan')           |                        8.8  |               79.4 |
| ('OOS 2024-26', 'B baseline (rs_126_skip21)') |                        4.04 |               63.3 |
| ('OOS 2024-26', 'R residual momentum')        |                        2.05 |               56.7 |
| ('OOS 2024-26', 'F frog-in-the-pan')          |                        1.59 |               43.3 |

|                             |   mean diff IS 2021-23 (pp) |   mean diff OOS 2024-26 (pp) |   t (full, non-overlapping) | adopt?   |
|:----------------------------|----------------------------:|-----------------------------:|----------------------------:|:---------|
| R residual momentum minus B |                       -2.46 |                        -2    |                       -1.34 | no       |
| F frog-in-the-pan minus B   |                       -5.97 |                        -2.46 |                       -4.57 | no       |
