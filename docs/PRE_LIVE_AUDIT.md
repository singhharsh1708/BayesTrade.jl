# Pre-live audit: verdict

Scope: everything between a live market feed and an order. Phases A through L of the hardening
brief. This is the go/no-go, and it stops short of "the tests are green" on purpose. A green
suite says the code does what the tests say. It says nothing about whether the tests describe a
system that should be given money.

## Verdict

**Paper trading against a live feed: go.** Every gate the paper path depends on has been
exercised, and each one has been mutated to confirm the test that guards it actually fails
without it.

**Real money: no-go, and not close.** Not because a check is failing. Because the code that
would send an order does not exist, three structural gaps below are open, and two phases of
this brief could not be run at all without live market data.

## What was found

Nine defects, five of them HIGH, all fixed in #28. The five that could have moved money:

| # | Severity | Defect | Why it mattered |
|---|---|---|---|
| 1 | HIGH | `Inf > 0` is true, so an infinite price passed the positivity check | It propagated into bars, features and posteriors without ever raising |
| 2 | HIGH | Volume accepted `NaN`, `Inf` and negatives | A `NaN` volume makes every downstream comparison false, which is the direction that trades |
| 3 | HIGH | Every `DateTime` was naive and Kite sends epoch seconds | `unix2datetime` yields UTC against a 09:15 IST open: every bar 5.5 hours out of place, the open at 03:45 |
| 4 | HIGH | The session's staleness guard was structurally dead | Accepting the tick that produced a bar refreshes the feed clock, so the check always saw a fresh feed. A session fed a one-year gap traded straight through it |
| 5 | HIGH | An unwritable journal took the session down | Being unable to explain a decision is a reason to stop, not to crash |

The remaining four: a foreign tick killed the session, book prices and quantities were
unvalidated, a halted account displayed ceiling checks it never reached, and a reducing trade
could be charged against headroom as though it were an opening one.

One thing expected to be a bug was not. Bars cannot span the overnight gap: buckets are absolute
timestamps floored to the interval, so two ticks on different dates never share one. The guard
written for it was removed and the property asserted across four intervals instead.

## What was verified by mutation

A test that passes is not evidence. Each protection was mutated back to the buggy behaviour to
confirm the guard fails without it.

| Protection | Mutation caught by |
|---|---|
| replay settles on `realised_at`, not arrival | 16 failures |
| replay refit window excludes the predicted bar | 10 failures |
| session settles by bar count, not calendar time | 2 failures |
| session scores against the forecast stamp | 1 failure |
| `features_at` cannot read past `as_of` | 169 failures |
| `walk_forward` embargo | 23 failures |
| slippage always against the order | caught |
| commission on both sides | caught |
| participation cap | caught |
| reducing trade not charged headroom | caught |
| `approved <= requested` | caught |

One risk-engine mutant survived and is an equivalent mutant rather than a hole: removing the
early return leaves the final `any(failed, checks)` in place, so the ruling is unchanged and only
the audit trail differs. That trail is now pinned by a test of its own.

## What answers "why did it buy here"

A single journal line reconstructs a decision with no models, no market and no process: the bar,
the reference price, the feature vector the models actually saw, every component posterior with
its weight and uncertainty beside the pooled one, the credible interval, all nine risk gates with
what each observed and allowed, and the fill with its slippage and fees.

Component posteriors sit beside the fused one deliberately. A fused number says what the system
believed; only the components say why, and which model was carrying the opinion is the first
thing anyone asks.

## What survives a restart

Measured against a process killed at bar 800 holding 36 fills:

```
naive restart:     fills=95   <- 59 duplicate orders
resumed restart:   fills=0  replayed=799
```

`read_journal` tolerates a torn trailing line, because a crash cuts the last write in half and
discarding weeks of good history over one truncated line is the worse failure.

## Open blockers

Ranked by what has to be true before real money is plausible, not by effort.

**1. There is no live broker, and that is the strongest gate in the system.** `subtypes(Broker)
== [PaperBroker]`, asserted by a test that fails the moment a second subtype appears. A real
order is not gated behind a flag; the code path is absent. Building it is a Phase 15 decision,
and building it removes this gate, so it should be the last thing done and not the first.

**2. Position state is not reconciled against the broker.** ~~The session believes its own book.~~
**Closed.** `reconcile` compares the local book against a venue snapshot and reports matched,
mismatched or unavailable; a mismatch or an unreadable venue stops new trading through
`check_health`, and nothing resolves automatically. `AccountSource` is a read-only interface, not
a broker, so this was built and tested with no live order path in existence.

What remains is the venue implementation itself, which arrives with the broker and not before.

**3. `resume!` does not rebuild the position book.** ~~It sets a watermark.~~ **Closed.** It now
replays every fill in the journal into positions and cash, checks the reconstruction against the
equity the journal last recorded, and installs it only if that holds. Where it does not, nothing
is installed, the watermark is still set, and `check_health` fails on `:state_rebuild`. Peak
equity is restored too, so the drawdown limit does not rearm at the restart.

**4. The calendar covers two years.** 2025 and 2026, each taken from independent published
sources that agreed on every date. **Still open, and deliberately.** At the time of writing the
NSE had not published 2027; projections of it circulate and none of them is the exchange. Adding
one would be exactly the unverified dataset this project has refused elsewhere.

What changed is that the refusal is now actionable: `CalendarCoverageError` names the year, the
standard the shipped years were held to, and the `load_calendar` call that closes it. Every
calendar must also carry its `sources`, shipped or supplied, because a list of dates with no
provenance is a list somebody typed.

## What could not be done

**Phase F, live feed and paper trading against a real market.** Requires a live market data
connection. Everything up to the socket is exercised against recorded and synthetic ticks: the
binary tick protocol byte by byte, aggregation, staleness, session boundaries, timezone
conversion. The socket itself has never been opened.

**Phase H, calibration against live data.** Requires the same. Calibration is exercised against
synthetic series with known generating processes, which tests the arithmetic and cannot test
whether this market's returns behave the way the models assume.

Neither can be closed by writing code, and reporting them as done because their unit tests pass
would be the exact failure this document exists to prevent.

## Standing conditions

1. `subtypes(Broker) == [PaperBroker]`. No implementation can reach a venue.
2. `PaperTradingSession.broker` is typed `PaperBroker`, so a live broker could not be installed
   there even if one existed.
3. A live `KiteSession` requires `BAYESTRADE_ALLOW_LIVE_TRADING=true` in addition to being asked
   for live mode. One variable set by accident is not consent.
4. `load_settings()` refuses a half-set live configuration.
5. The Groww client can build exactly two request paths, and neither is an order endpoint. The
   refusal happens when the request is assembled, not when it is sent.
6. `check_health` runs before every action, and every check is written so that being unable to
   measure its input fails. A check that passes when it cannot see turns itself off exactly when
   something has gone wrong.
