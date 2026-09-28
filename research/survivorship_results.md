# Survivorship bias: today's universe vs a point-in-time universe

Design and limitations: see the docstring of research/survivorship_study.py.
65 rebalance dates; 603 candidate companies (459 with NSE prices); 101 daily moves beyond +/-35% treated as splits/bonuses.
Eligible stocks per date (after the trend and history rules): survivor 61, point in time 48, of which 4 are not in today's universe (big then, small now, or illiquid then).

|                                                  |   mean fwd RS of top 8 (pp) |   beat Nifty 500 % |   rank IC |   Q5-Q1 spread (pp) |
|:-------------------------------------------------|----------------------------:|-------------------:|----------:|--------------------:|
| ('All', "S survivor (today's universe)")         |                        8.95 |               75.4 |     0.053 |                3.77 |
| ('All', 'P point in time')                       |                        5.83 |               69.2 |     0.071 |                5.45 |
| ('IS 2021-23', "S survivor (today's universe)")  |                       14.29 |               91.4 |     0.071 |                6.49 |
| ('IS 2021-23', 'P point in time')                |                        9.93 |               85.7 |     0.081 |                6.88 |
| ('OOS 2024-26', "S survivor (today's universe)") |                        2.72 |               56.7 |     0.033 |                0.6  |
| ('OOS 2024-26', 'P point in time')               |                        1.04 |               50   |     0.059 |                3.84 |

Top 8, point in time minus survivor: -3.12 pp per quarter (t = -1.92, non-overlapping quarters).
