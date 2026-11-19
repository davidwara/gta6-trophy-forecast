"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Reads one case (makefun/gta6-trophies-case/v1): the house recording of
Exophase's PS5 trophy page for Grand Theft Auto VI up to a checkpoint, and
writes five quantiles (5/25/50/75/95%) of every trophy's earn rate at the
case's target, as /output/<id>.out. Standard library only, offline,
deterministic: the same case always yields the same bytes.

Why this shape, and what the numbers mean
------------------------------------------
An Exophase earn rate is a ratio, r = E / N: E trophy earners among N
tracked players. In a launch week both move, and they move for different
reasons. N grows as new players are tracked; E grows as the players who
are already there play further into the game. A rate can therefore fall
even while more people earn the trophy, whenever N grows faster than E,
which is exactly what the launch-night refresh showed (638 tracked players
became 3,868 and almost every rate fell). Fitting the rate directly, as a
trend line on the rate, confuses the two motions. So this program fits
the two counts separately and divides at the end:

* N(t): log N against log(1 + hours since release), least squares over the
  distinct page states in the case's history. A power law in launch age
  is the honest shape for a tracked-player base that grows fast at first
  and then slows; its slope is clamped so a short, noisy window cannot
  explode the forecast.
* E_i(t): the same log-log fit per trophy, anchored at its last observed
  count (the level is taken from the data, only the slope is fitted), with
  the slope shrunk toward the median slope of its group, because a trophy
  with four earners cannot identify its own slope. Earners never fall:
  the forecast count is floored at the last observed count.

On top of the fit sits a launch-curve prior. A trophy's week-one rate
follows a saturating curve in launch age, m(T) = cap * (T / (T + K)) ** p,
with a cap, a half-life K and a shape p per group of trophies: story
chapters climb high and early, the platinum and 100% completion crawl,
grinds sit in between. The prior is evaluated at the target's launch age,
so for a 90-minute practice horizon it stays near the opening rates and
for the 6,510-minute final horizon it expects real progress. The fitted
forecast, the prior and the last observed rate are blended in logit space
with horizon-dependent weights: the shorter the horizon, the more the
last rate anchors the answer; the more distinct page states the history
holds, the more the fit is trusted.

Spreads are honest about what can go wrong. The base spread grows with
the square root of the horizon, like a random walk on the logit scale.
On top of that come the fit's residual noise, Poisson noise on small
earner counts, and a staleness term: Exophase refreshes its counts on its
own schedule, so a page that has sat unchanged for far longer than its
usual refresh gap may jump at the next refresh, and a history with only
one distinct state has never shown a refresh at all. Quantiles are built
from the blended median plus a normal spread on the logit scale, then
clamped to [0.01, 99.99] and rounded to four decimals.

Two structural facts are enforced after blending. Story chapters are
earned in order, so their medians are monotone (chapter 1 >= chapter 2
>= ... >= final mission), repaired with pool-adjacent-violators on the
logit medians. And the platinum needs every other trophy, so its median
never exceeds the smallest other median or the 100%-completion median.
A trophy every tracked player holds (100% in every read, the opening
mission) keeps a near-100 forecast with only a hair of doubt at the 5%
quantile: its rate is structural, not a trend.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Launch-curve prior per trophy group: (cap_percent, half_life_hours, shape).
# Story chapters climb highest; the platinum and 100% crawl; grinds sit low.
STORY_PRIOR = {
    3: (45.0, 72.0, 1.5),   # Chapter 1
    4: (38.0, 96.0, 1.5),   # Chapter 2
    5: (32.0, 120.0, 1.5),  # Chapter 3
    6: (27.0, 144.0, 1.5),  # Chapter 4
    7: (20.0, 168.0, 1.6),  # Final mission
}
SIDE_PRIOR = (16.0, 150.0, 1.5)       # Side-story mission strands
PLATINUM_PRIOR = (1.0, 300.0, 2.0)    # Platinum: needs everything
COMPLETION_PRIOR = (1.5, 300.0, 2.0)  # 100% completion
GRIND_PRIOR = (4.0, 240.0, 1.8)       # Long grinds: stunts, cards, WAiNK, challenges
SKILL_PRIOR = (6.0, 200.0, 1.7)       # Skill gates: gold medals, races, vaults
HARD_MISC_PRIOR = (5.0, 200.0, 1.7)   # Slow misc: explore all, big spends
EASY_MISC_PRIOR = (10.0, 150.0, 1.5)  # Quick misc: arcade, fishing, cocktails

GRIND_IDS = {15, 19, 28, 30, 31, 32, 42, 52, 54}
SKILL_IDS = {17, 18, 25, 40}
SIDE_IDS = {8, 9, 10, 11, 12, 13}
CHAPTER_IDS = [3, 4, 5, 6, 7]


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (no datetime import needed)."""
    date_part, _, time_part = text.partition("T")
    time_part = time_part.rstrip("Z")
    if "." in time_part:
        time_part = time_part.split(".")[0]
    year, month, day = (int(v) for v in date_part.split("-"))
    hour, minute, second = (int(v) for v in time_part.split(":"))
    days = _days_from_civil(year, month, day)
    return days * 86400 + hour * 3600 + minute * 60 + second


def _days_from_civil(y, m, d):
    """Days since 1970-01-01 (Howard Hinnant's civil-from-days inverse)."""
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when the fit is degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _prior_for(trophy_id, launch_age_hours):
    """Saturating launch-curve prior rate (percent) at a launch age."""
    if trophy_id in STORY_PRIOR:
        cap, half, shape = STORY_PRIOR[trophy_id]
    elif trophy_id == 1:
        cap, half, shape = PLATINUM_PRIOR
    elif trophy_id == 14:
        cap, half, shape = COMPLETION_PRIOR
    elif trophy_id in SIDE_IDS:
        cap, half, shape = SIDE_PRIOR
    elif trophy_id in GRIND_IDS:
        cap, half, shape = GRIND_PRIOR
    elif trophy_id in SKILL_IDS:
        cap, half, shape = SKILL_PRIOR
    elif trophy_id in {16, 20, 21, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                       45, 46, 47, 48, 49, 50, 51, 53, 55}:
        cap, half, shape = EASY_MISC_PRIOR
    else:
        cap, half, shape = HARD_MISC_PRIOR
    frac = launch_age_hours / (launch_age_hours + half)
    return cap * (frac ** shape)


def _group_of(trophy_id):
    if trophy_id in STORY_PRIOR:
        return ("story", trophy_id)
    if trophy_id == 1:
        return ("platinum", 0)
    if trophy_id == 14:
        return ("completion", 0)
    if trophy_id in SIDE_IDS:
        return ("side", 0)
    if trophy_id in GRIND_IDS:
        return ("grind", 0)
    if trophy_id in SKILL_IDS:
        return ("skill", 0)
    return ("misc", 0)


def _distinct_states(case, release_epoch):
    """Collapse consecutive identical reads; return list of state dicts.

    Each state: (hours_since_release, players_tracked, [rates]).
    Exophase refreshes in steps, so long flat runs are one state, not many.
    """
    states = []
    prev_key = None
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            at = _parse_time(read["at"])
        except (KeyError, ValueError, TypeError):
            continue
        hours = max(0.0, (at - release_epoch) / 3600.0)
        states.append({"hours": hours, "players": read.get("players_tracked") or 0,
                       "rates": rates})
    return states


def _recent_window(states, span_hours=36.0, minimum=3):
    """The states a slope should be fitted on: the recent window only.

    Launch week bends every curve: the first hours are dominated by
    pre-release players and the first refresh's dilution, while the
    recent day shows the steady motion that extrapolates. Fitting the
    whole history averages the bend in and understates the recent slope.
    """
    if not states:
        return []
    cutoff = states[-1]["hours"] - span_hours
    window = [s for s in states if s["hours"] >= cutoff]
    if len(window) < minimum:
        window = states[-minimum:] if len(states) >= minimum else list(states)
    return window


def _fit_players(states, target_hours):
    """Forecast players tracked at the target from the log-log fit."""
    if not states:
        return None, 0.0
    last_n = states[-1]["players"]
    if last_n <= 0:
        return None, 0.0
    window = _recent_window(states)
    xs = [math.log1p(s["hours"]) for s in window]
    ys = [math.log(max(1, s["players"])) for s in window]
    slope = _ols_slope(xs, ys)
    slope = min(1.3, max(0.0, slope))
    anchor_x = math.log1p(states[-1]["hours"])
    anchor_y = math.log(max(1, last_n))
    pred_log = anchor_y + slope * (math.log1p(target_hours) - anchor_x)
    pred = math.exp(pred_log)
    # Players never fall, and a 4.5-day jump cannot credibly 30x the base.
    pred = min(pred, last_n * 30.0)
    pred = max(pred, float(last_n))
    resid = 0.0
    if len(window) >= 3:
        mean_y = sum(ys) / len(ys)
        mean_x = sum(xs) / len(xs)
        icept = mean_y - slope * mean_x
        resid = math.sqrt(sum((y - (icept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / len(xs))
    return pred, resid


def _fit_earners(states, idx, target_hours, group_slope):
    """Forecast a trophy's earner count at the target (anchored log-log)."""
    points = []
    for s in _recent_window(states):
        rates = s["rates"]
        if idx >= len(rates) or rates[idx] is None:
            continue
        count = rates[idx] * max(1, s["players"]) / 100.0
        points.append((s["hours"], count))
    if not points:
        return None, 0.0, 0.0
    last_count = None
    for s in reversed(states):
        if idx < len(s["rates"]) and s["rates"][idx] is not None:
            last_count = s["rates"][idx] * max(1, s["players"]) / 100.0
            break
    if last_count is None:
        last_count = points[-1][1]
    if len(points) < 2:
        return last_count, 0.0, 0.0
    xs = [math.log1p(h) for h, _ in points]
    ys = [math.log(c + 0.5) for _, c in points]
    slope = _ols_slope(xs, ys)
    slope = min(1.6, max(0.0, slope))
    # Shrink a thin trophy's slope toward its group's median slope, unless
    # the caller asked for the raw slope (group_slope None: the pre-pass
    # that measures the groups in the first place).
    if group_slope is not None:
        weight = len(points)
        slope = (weight * slope + 3.0 * group_slope) / (weight + 3.0)
    last_x = math.log1p(states[-1]["hours"])
    last_y = math.log(last_count + 0.5)
    pred_log = last_y + slope * (math.log1p(target_hours) - last_x)
    pred = math.exp(pred_log) - 0.5
    resid = 0.0
    if len(points) >= 3:
        mean_y = sum(ys) / len(ys)
        mean_x = sum(xs) / len(xs)
        icept = mean_y - slope * mean_x
        resid = math.sqrt(sum((y - (icept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / len(xs))
    return max(pred, last_count), slope, resid


def _pool_adjacent_violators(values):
    """Make a list non-increasing (logit medians of the story chain)."""
    levels = [[v, 1] for v in values]
    out = []
    for v, w in levels:
        out.append([v, w])
        while len(out) >= 2 and out[-2][0] < out[-1][0]:
            total_w = out[-2][1] + out[-1][1]
            avg = (out[-2][0] * out[-2][1] + out[-1][0] * out[-1][1]) / total_w
            out[-2:] = [[avg, total_w]]
    result = []
    for v, w in out:
        result.extend([v] * w)
    return result


def forecast_case(case):
    trophies = case.get("trophies") or []
    release = (case.get("game") or {}).get("release_date") or "2026-11-19"
    release_epoch = _parse_time(release + "T00:00:00Z")
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    target_epoch = _parse_time(case["target"])
    target_hours = max(0.0, (target_epoch - release_epoch) / 3600.0)
    checkpoint_epoch = _parse_time(case["at"])
    states = _distinct_states(case, release_epoch)
    n_states = len(states)

    # Quality of the fit: more distinct states, more trust in the slopes.
    quality = 0.0 if n_states < 2 else min(1.0, (n_states - 1) / 5.0)
    w_last = 0.05 + 0.5 * math.exp(-horizon_days / 1.5)
    w_fit = 0.10 + 0.45 * quality
    w_prior = max(0.0, 1.0 - w_last - w_fit)

    # How stale is the last page state, versus the usual refresh gap?
    stale_extra = 0.0
    if n_states <= 1:
        stale_extra = 1.0  # Never saw a refresh: the next one can jump far.
    elif n_states >= 3:
        gaps = []
        for a, b in zip(states, states[1:]):
            gaps.append(max(0.0, b["hours"] - a["hours"]))
        typical_gap = _median(gaps) if gaps else 0.0
        stale_hours = max(0.0, (checkpoint_epoch - release_epoch) / 3600.0
                          - states[-1]["hours"])
        if typical_gap > 0 and stale_hours > 2.0 * typical_gap:
            stale_extra = 0.4

    pred_n, n_resid = _fit_players(states, target_hours)

    # Group slopes first, so thin trophies can borrow their group's motion.
    raw_slopes = {}
    for i in range(len(trophies)):
        _, slope, _ = _fit_earners(states, i, target_hours, None)
        if slope is not None:
            raw_slopes[i] = slope
    group_slopes = {}
    for i in range(len(trophies)):
        grp = _group_of(trophies[i]["id"])
        members = [raw_slopes[j] for j in range(len(trophies))
                   if _group_of(trophies[j]["id"]) == grp and j in raw_slopes]
        group_slopes[grp] = _median(members) if members else 0.0

    last_rates = []
    if states:
        last_rates = states[-1]["rates"]
    rows = []
    medians_logit = {}
    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        last_rate = None
        if i < len(last_rates) and last_rates[i] is not None:
            last_rate = last_rates[i]
        if last_rate is None:
            # Never observed: fall back to the most recent non-null read.
            for s in reversed(states):
                if i < len(s["rates"]) and s["rates"][i] is not None:
                    last_rate = s["rates"][i]
                    break
        if last_rate is None:
            last_rate = 0.5

        # A trophy everyone holds in every read is structural, not a trend.
        if states and all(
                i < len(s["rates"]) and s["rates"][i] == 100 for s in states):
            rows.append({"id": tid, "q": [99.9, 99.99, 99.99, 99.99, 99.99]})
            medians_logit[tid] = _logit(99.99)
            continue

        group_slope = group_slopes.get(_group_of(tid), 0.0)
        pred_e, _, e_resid = _fit_earners(states, i, target_hours, group_slope)
        if pred_e is None or pred_n is None or pred_n <= 0:
            rate_fit = last_rate
        else:
            rate_fit = 100.0 * pred_e / pred_n
        rate_fit = _clamp_pct(rate_fit)

        # Dilution floor: earners never fall, so the rate cannot drop below
        # the last earners divided by the forecast tracked base.
        if states and pred_n:
            last_count = last_rate * max(1, states[-1]["players"]) / 100.0
            floor_rate = 100.0 * last_count / pred_n
        else:
            floor_rate = 0.0
        rate_prior = max(_prior_for(tid, target_hours), floor_rate * 0.5)
        rate_prior = _clamp_pct(rate_prior)

        blended = (w_fit * _logit(rate_fit) + w_prior * _logit(rate_prior)
                   + w_last * _logit(last_rate))
        if n_states <= 1:
            # A page frozen since launch night has never shown a refresh,
            # yet the real player base behind it is exploding: the next
            # refresh multiplies the tracked base several-fold (launch
            # night: 638 -> 3,868 tracked) while earners grow far slower,
            # so every rate falls. A ~4.5x dilution is -1.5 logit units.
            # This only fires when the history holds a single page state;
            # a case with real refreshes in its history never sees it.
            blended -= 1.5
        medians_logit[tid] = blended

        # Spread: random-walk base, fit noise, small-count noise, staleness.
        sigma = 0.45 * math.sqrt(horizon_days)
        sigma = math.sqrt(sigma * sigma + (0.5 * min(e_resid, 0.8)) ** 2
                          + (0.5 * min(n_resid, 0.8)) ** 2
                          + stale_extra * stale_extra)
        if last_rate is not None and states:
            earners = last_rate * max(1, states[-1]["players"]) / 100.0
            sigma = math.sqrt(sigma * sigma
                              + (0.5 / math.sqrt(max(1.0, earners))) ** 2)
        sigma = min(1.6, max(0.3, sigma))

        quantiles = []
        prev = None
        for z in Z:
            q = _round4(_clamp_pct(_sigmoid(blended + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": tid, "q": quantiles})

    # Story chapters in order: repair any monotone violation in the medians.
    by_id = {row["id"]: row for row in rows}
    chain = [tid for tid in CHAPTER_IDS if tid in by_id]
    if len(chain) >= 2:
        fixed = _pool_adjacent_violators([medians_logit[t] for t in chain])
        for tid, new_logit in zip(chain, fixed):
            if abs(new_logit - medians_logit[tid]) < 1e-9:
                continue
            row = by_id[tid]
            old_median = row["q"][2]
            shift = new_logit - medians_logit[tid]
            new_q = []
            prev = None
            for q in row["q"]:
                moved = _round4(_clamp_pct(_sigmoid(_logit(q) + shift)))
                if prev is not None and moved < prev:
                    moved = prev
                new_q.append(moved)
                prev = moved
            row["q"] = new_q
            medians_logit[tid] = new_logit
    # Platinum needs every trophy: never above the smallest other median.
    if 1 in by_id and len(rows) > 1:
        others = [medians_logit[t] for t in medians_logit if t != 1]
        cap_logit = min(others) if others else medians_logit[1]
        if 14 in medians_logit:
            cap_logit = min(cap_logit, medians_logit[14])
        if medians_logit[1] > cap_logit:
            row = by_id[1]
            shift = cap_logit - medians_logit[1]
            new_q = []
            prev = None
            for q in row["q"]:
                moved = _round4(_clamp_pct(_sigmoid(_logit(q) + shift)))
                if prev is not None and moved < prev:
                    moved = prev
                new_q.append(moved)
                prev = moved
            row["q"] = new_q
            medians_logit[1] = cap_logit
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
