# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/davidwara/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/davidwara/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI trophy earn rates on Exophase, built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC. Given the house recording of the trophy page up to a checkpoint, it predicts every trophy's earn rate at the target as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

How it forecasts: an Exophase rate is earners over tracked players, and in launch week the two move for different reasons, so the program fits them separately — log-log power-law fits of the tracked-player base and of each trophy's earner count over the recent window, shrunk toward group slopes — then divides. That fit is blended in logit space with a saturating launch-curve prior per trophy group (story chapters climb high, the platinum crawls) and with the last observed rate, with horizon-dependent weights. Spreads grow with the horizon and with page staleness, because Exophase refreshes on its own schedule and a frozen page can jump. Story chapters are kept monotone and the platinum is capped by the trophies it needs. Standard library only, offline, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=z5073zsc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
