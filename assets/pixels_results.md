| method | params | solved | steps to solve | episodes to solve | time to solve | final greedy return | episodes >= 475 | steps run | wall time |
|---|---|---|---|---|---|---|---|---|---|
| `ppo [frame, cnn]` | 330,211 | no | - | - | - | 16.6 +- 9.3 (20 ep.) | 0% | 50,176 | 272 s |
| `reinforce-baseline [frame-diff, cnn]` | 330,611 | no | - | - | - | 37.0 +- 13.7 (20 ep.) | 0% | 50,082 | 315 s |
| `ppo [frame-diff, cnn]` | 330,611 | no | - | - | - | 27.1 +- 10.6 (20 ep.) | 0% | 50,176 | 301 s |
| `ppo [frame-diff, rescnn]` | 335,251 | no | - | - | - | 32.7 +- 15.2 (20 ep.) | 0% | 50,176 | 346 s |
