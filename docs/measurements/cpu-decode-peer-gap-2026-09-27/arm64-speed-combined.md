| model | pass 1 new / old (new first) | pass 2 new / old (old first) | pass 1 new ÷ old | pass 2 new ÷ old | **combined (geo-mean)** | Ollama p1 / p2 | new ÷ Ollama (mean) | old ÷ Ollama (mean) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5B | 57.3 / 106.2 | 58.6 / 107.0 | 0.540× | 0.548× | **0.544×** | 130.8 / 147.3 | 0.417× | 0.768× | slower in both passes: NEON-widen follow-up flagged |
| 1.5B | 27.5 / 50.6 | 28.6 / 50.6 | 0.543× | 0.565× | **0.554×** | 62.9 / 71.7 | 0.418× | 0.753× | slower in both passes: NEON-widen follow-up flagged |
| 7B | 7.2 / 17.0 | 7.9 / 17.7 | 0.424× | 0.446× | **0.435×** | 16.5 / 18.2 | 0.435× | 1.001× | slower in both passes: NEON-widen follow-up flagged |

pass 1: 9 cells; errors / non-ok token gates: none

pass 2: 9 cells; errors / non-ok token gates: none
