# Stop-loss and replacement rules vs holding

Design and limitations: see the docstring of research/stop_rule_study.py.
65 rebalance dates, point-in-time universe, 0.25% cost per trade.

|                                                      |   mean 3m RS vs Nifty 500 (pp) |   beat Nifty 500 % |   worst quarter (pp) |   stock exits per quarter |
|:-----------------------------------------------------|-------------------------------:|-------------------:|---------------------:|--------------------------:|
| ('All', 'H hold (no stops)')                         |                           5.83 |               69.2 |               -18.01 |                       0   |
| ('All', 'S stops, cash')                             |                           1.78 |               52.3 |               -17.32 |                       6.6 |
| ('All', 'R stops + replacement (live rule)')         |                           2.54 |               50.8 |               -19.32 |                       8.3 |
| ('All', 'R 4xATR trailing')                          |                           4.17 |               63.1 |               -16.28 |                       5.4 |
| ('All', 'R 5xATR trailing')                          |                           4.46 |               60   |               -11.6  |                       3.4 |
| ('All', 'R 6xATR trailing')                          |                           4.96 |               64.6 |               -14.86 |                       2.1 |
| ('All', 'R 3xATR fixed')                             |                           4.11 |               64.6 |               -19.32 |                       3.8 |
| ('All', 'R 4xATR fixed')                             |                           4.41 |               63.1 |               -16.73 |                       2.3 |
| ('All', 'R 5xATR fixed')                             |                           5.2  |               66.2 |               -13.69 |                       1.4 |
| ('IS 2021-23', 'H hold (no stops)')                  |                           9.93 |               85.7 |               -13.79 |                       0   |
| ('IS 2021-23', 'S stops, cash')                      |                           4.65 |               62.9 |               -14.28 |                       6   |
| ('IS 2021-23', 'R stops + replacement (live rule)')  |                           6.42 |               62.9 |               -11.6  |                       7.4 |
| ('IS 2021-23', 'R 4xATR trailing')                   |                           8.52 |               82.9 |                -9.19 |                       4.6 |
| ('IS 2021-23', 'R 5xATR trailing')                   |                           8.96 |               80   |                -5.71 |                       2.8 |
| ('IS 2021-23', 'R 6xATR trailing')                   |                           9.71 |               82.9 |                -8.7  |                       1.5 |
| ('IS 2021-23', 'R 3xATR fixed')                      |                           8.71 |               85.7 |               -13.12 |                       2.8 |
| ('IS 2021-23', 'R 4xATR fixed')                      |                           9.07 |               82.9 |               -13.12 |                       1.6 |
| ('IS 2021-23', 'R 5xATR fixed')                      |                           9.62 |               82.9 |               -12.64 |                       0.9 |
| ('OOS 2024-26', 'H hold (no stops)')                 |                           1.04 |               50   |               -18.01 |                       0   |
| ('OOS 2024-26', 'S stops, cash')                     |                          -1.56 |               40   |               -17.32 |                       7.2 |
| ('OOS 2024-26', 'R stops + replacement (live rule)') |                          -1.98 |               36.7 |               -19.32 |                       9.3 |
| ('OOS 2024-26', 'R 4xATR trailing')                  |                          -0.9  |               40   |               -16.28 |                       6.4 |
| ('OOS 2024-26', 'R 5xATR trailing')                  |                          -0.78 |               36.7 |               -11.6  |                       4.2 |
| ('OOS 2024-26', 'R 6xATR trailing')                  |                          -0.57 |               43.3 |               -14.86 |                       2.8 |
| ('OOS 2024-26', 'R 3xATR fixed')                     |                          -1.26 |               40   |               -19.32 |                       4.9 |
| ('OOS 2024-26', 'R 4xATR fixed')                     |                          -1.02 |               40   |               -16.73 |                       3.1 |
| ('OOS 2024-26', 'R 5xATR fixed')                     |                           0.03 |               46.7 |               -13.69 |                       2   |

|                                           |   mean diff IS 2021-23 (pp) |   mean diff OOS 2024-26 (pp) |   t (full, non-overlapping) |
|:------------------------------------------|----------------------------:|-----------------------------:|----------------------------:|
| S stops, cash minus H                     |                       -5.28 |                        -2.61 |                       -3.93 |
| R stops + replacement (live rule) minus H |                       -3.51 |                        -3.02 |                       -3.44 |
| R 4xATR trailing minus H                  |                       -1.41 |                        -1.95 |                       -2.14 |
| R 5xATR trailing minus H                  |                       -0.97 |                        -1.82 |                       -2.81 |
| R 6xATR trailing minus H                  |                       -0.22 |                        -1.61 |                       -2.21 |
| R 3xATR fixed minus H                     |                       -1.22 |                        -2.3  |                       -2.81 |
| R 4xATR fixed minus H                     |                       -0.86 |                        -2.06 |                       -3.28 |
| R 5xATR fixed minus H                     |                       -0.31 |                        -1.01 |                       -2.7  |
