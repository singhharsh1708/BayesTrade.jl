# Runbook

Operating BayesTrade in paper mode. Real-money trading is not covered because it is not
possible: see [Safety gates](#safety-gates).

## Startup

Run in order. Each step is a gate, not a formality.

**1. Verify configuration.**

```sh
julia --project=. -e 'using BayesTrade; s = load_settings(); println(s)'
```

Expect `trading_mode = PAPER`. If it prints anything else, stop and find out who set
`BAYESTRADE_TRADING_MODE`.

**2. Verify trading mode is paper.**

```sh
env | grep -E 'BAYESTRADE_(TRADING_MODE|ALLOW_LIVE_TRADING)' || echo "unset, which is the default and correct"
```

Both unset is the safe state. `load_settings()` refuses to construct a live configuration
unless both are set and agree.

**3. Verify the broker.**

The paper broker needs nothing. It starts with the cash it is given and keeps its own
positions. There is no venue to reach.

**4. Verify market data.**

```sh
julia --project=. examples/backtest.jl edge
```

A successful run prints a calibration report. This exercises the whole pipeline on generated
data and proves the installation works before any live feed is involved.

**5. Verify the risk limits in force.**

```sh
julia --project=. -e 'using BayesTrade; println(RiskLimits())'
```

Read them. They are the only thing standing between a model's opinion and a position, and they
are the record an audit will re-derive a past ruling from.

**6. Start paper trading.**

```sh
julia --project=. examples/paper_session.jl 1400 journal.jsonl
```

**7. Confirm it is healthy before believing anything it does.**

```julia
println(summarise(check_health(session, last_bar_time)))
```

`HALTED` means it is not trading and will tell you which condition failed.

## Normal shutdown

Stop feeding ticks, then close the open bar so the last partial candle is not lost:

```julia
close_bar!(session)
```

The journal is already on disk: every line was appended as it happened, so nothing is buffered
waiting for a clean exit.

## Emergency shutdown

Run in this order. The order is the procedure: each step is only safe once the one above it has
happened.

### 1. Stop new decisions

```julia
session.halted = true            # if the session exposes it, else stop feeding
```

Stop feeding ticks. Nothing decides on a bar it never sees, and this is the only step that is
instant and cannot fail.

### 2. Stop new orders

In this build there is nothing to stop: `subtypes(Broker) == [PaperBroker]` and no code can reach
a venue. **When a live broker exists**, cancel every working order before anything else, and record
the cancel acknowledgements. An order left working during a shutdown fills while nobody is
watching, and the position it creates is not in any book.

### 3. Persist state

```julia
close_bar!(session)              # the open bar, so the last partial candle is not lost
```

The journal is already on disk: every line was appended as it happened and nothing is buffered
waiting for a clean exit. `close_bar!` is the only thing that can still be lost, and only the
partial bar.

### 4. Reconcile

```julia
result = reconcile!(session, venue)
println(reconciliation_report(result))
```

Before the process ends, while the venue is still reachable. A reconciliation run after the fact
compares the local book against a venue that has moved on.

If it does not match, **write down what it said** and do not act on it yet. Section
[Broker and API failure](#broker-and-api-failure) is the procedure.

### 5. Alert

The journal is the alert. Every halt, every failed write, every reconciliation is a line in it:

```sh
grep -E '"event":"(halted|reconciliation)"' journal.jsonl | tail -20
```

There is no paging integration and this document does not pretend otherwise. A deployment that
needs one has to add it.

### What happens to each thing

| | on emergency shutdown |
|---|---|
| **open positions** | left open. Nothing here liquidates: a forced unwind into a market that is already misbehaving is how a bad afternoon becomes a bad quarter. They are reconciled and handed to an operator. |
| **pending orders** | cancelled first, in step 2, before anything else. In this build there are none. |
| **paper orders** | discarded with the process. They were never anywhere else. |
| **future orders** | none exist. There is no scheduling or queueing of orders in this system, by design. |
| **model state** | discarded. It is refitted from bars on restart, deterministically, and nothing about it is worth preserving across a failure. |
| **journal state** | already durable. Append-only, flushed per line, and a torn final line is tolerated on read. |

## Broker and API failure

Paper mode cannot produce most of these: the broker is in-process. Each is written for the live
broker that does not exist yet, and the entry conditions are the ones the code already
distinguishes.

### The rule that governs all of them

**Trading stays disabled until the books agree.** Not until the error clears, not until the
connection returns. Those are different events, and the second one is the one that matters.

```julia
may_open_new_positions(session.reconciliation)   # false unless RECONCILE_MATCHED
```

### Timeout on a request

The order may have reached the venue and the acknowledgement may have been lost. **Do not resend.**
A resend is how one intended position becomes two.

1. Stop sending new orders.
2. Reconcile. The venue's position and working-order list is the only thing that says whether the
   order arrived.
3. If the venue shows the order, it arrived: adopt nothing, record the discrepancy, and let an
   operator decide.
4. If the venue does not show it and the position is unchanged, it did not arrive.

### Authentication failure

The token expired, was revoked, or the credentials were rotated. Reconciliation reports
`RECONCILE_UNAVAILABLE`, which is not a match, so trading is already stopped.

Re-authenticate, reconcile, and only then resume. **Do not resume on a successful login alone**: a
new token says the venue is reachable and says nothing about what happened while it was not.

### Connection failure

Same as authentication failure in every respect that matters. The account could not be read,
`RECONCILE_UNAVAILABLE` is not a match, trading is stopped. Reconcile on reconnection before
anything else.

### Rejected order

The venue said no. The local book must not have assumed it: an order that is rejected has to leave
no trace in the position book, which `test_rebuild.jl` asserts.

Read the rejection reason from the receipt. A rejection for insufficient margin is an account
problem and a rejection for a bad symbol is a configuration problem, and they need different
people.

### Unknown order status

The worst of them, and the reason reconciliation exists. The order exists somewhere and this
system does not know its state.

**Never resend. Never assume filled. Never assume rejected.** Reconcile, and if the venue cannot
say either, trading stays disabled until a human establishes what happened by another route.

### Partial fill

Ordinary and not a failure. The journal records what filled rather than what was asked for, so the
book is correct by construction and `rebuild_account` replays it as-is.

The failure case is a partial fill followed by a timeout on the remainder, which is *Unknown order
status* above.

### Duplicate submission risk

The two ways one intended trade becomes two:

- **resending after a timeout**, which the timeout procedure forbids
- **a restart re-processing bars already acted on**, which the watermark in `resume!` prevents and
  `test_rebuild.jl` measures: a naive restart produced 95 fills where a resumed one produced 0

Order identifiers are generated locally and recorded in the journal, so a duplicate is detectable
after the fact even where it could not be prevented.

### Reconciliation procedure

```julia
result = reconcile!(session, venue)
println(reconciliation_report(result))
```

On a mismatch:

1. **Do not adopt the venue's numbers.** A quantity differing by one lot could be a missed fill, a
   duplicate fill, or a manual trade. Overwriting the local book erases the evidence of which.
2. Establish which side is wrong, from the venue's own fill history and the journal.
3. Correct the cause, not the symptom.
4. Reconcile again. Trading resumes only on `RECONCILE_MATCHED`.

### When trading must remain disabled

Until **all** of these are true:

- reconciliation returns `RECONCILE_MATCHED`
- no order is in an unknown state
- `check_health` returns tradeable
- an operator has read the journal lines covering the incident

## Restart

```julia
session = PaperTradingSession(...)      # a fresh session, same configuration
state = resume!(session, "journal.jsonl")
```

### What resume does, in order

1. **Reads the journal** and takes the watermark: the timestamp of the last bar acted on.
2. **Replays every fill** into positions and cash. The opening balance comes from the
   `session_started` line, which is why the constructor writes one.
3. **Checks the reconstruction** before believing it:
   - an opening balance must be present
   - the equity implied by the replay must agree with the equity the journal last recorded, within
     `REBUILD_EQUITY_TOLERANCE`
   - every fill must carry a price
   - one torn final line is an ordinary crash; several unreadable lines are not
4. **Installs the positions** only if all of that held, along with the peak equity so the drawdown
   limit does not rearm at the restart.
5. **Sets the watermark** either way, so no bar is traded twice even when the book could not be
   rebuilt.

### Verifying it

```julia
println(rebuild_report(session.rebuild))
println(session_report(session))
```

Check three things: the position count matches what the previous process held, the cash balance
matches, and `replayed` climbs as the feed catches up rather than `fills` climbing.

### If it cannot be reconstructed

Nothing is installed and `check_health` fails on `:state_rebuild`. **This is the correct outcome
and must not be worked around.** A confident wrong book is worse than an empty one: the empty book
trades nothing until somebody looks, and the wrong one sizes every decision against a position
that is not there.

Reconcile against the venue, which is the authority, and resume from what it says once a human has
established the true state.

### With a live broker

Reconcile before the first decision, not after. `resume!(session, path; install_positions = false)`
rebuilds without installing, for deployments that want the venue to be the only authority.

## Fail-closed conditions

Every condition under which this system refuses to make a new trade. Each is enforced in code, not
by convention.

| Condition | Where |
|---|---|
| the feed has been silent longer than `max_silence` | `check_health`, `:feed` |
| a tick arrived after a gap longer than `max_silence` | `check_health`, `:feed` |
| any model is unfitted | `check_health`, `:models` |
| model state is not usable | `check_health`, `:model_state` |
| the account state cannot be read | `check_health`, `:account` |
| it is not a trading day | `check_health`, `:trading_day` |
| the journal could not be written | `check_health`, `:journal` |
| settlement is behind | `check_health`, `:settlement` |
| a restart could not rebuild the position book | `check_health`, `:state_rebuild` |
| reconciliation did not match | `check_health`, `:reconciliation` |
| the venue could not be read | `check_health`, `:reconciliation` |
| trading is halted by the kill switch | `review`, `:kill_switch` |
| the daily loss limit is breached | `review`, `:daily_loss` |
| the drawdown limit is breached | `review`, `:drawdown` |
| the open-position count is at its limit | `review`, `:open_positions` |
| annualised volatility is above its limit | `review`, `:volatility` |
| turnover is below the liquidity floor | `review`, `:liquidity` |
| no headroom under a position, sector or portfolio ceiling | `review` |
| the predicted edge is too small | `decide`, `edge_too_small` |
| P(up) is below its floor | `decide`, `probability_positive` |
| P(large loss) is above its ceiling | `decide`, `probability_large_loss` |
| model uncertainty is above its ceiling | `decide`, `predictive_sd` |
| too few models answered | `decide`, `n_models` |
| the calendar cannot answer for this year | `CalendarCoverageError` |
| a price or volume is not finite | `Bar` and `Quote` constructors |
| the model health score is unacceptable | `assess_model_health` |

Two of these are worth stating separately because they are easy to misread.

**A gate that cannot measure its input fails.** Not passes. This is the whole design of
`check_health` and `assess_model_health`: a check that passes when it cannot see turns itself off
exactly when something has gone wrong.

**A skipped risk check is not a passed one.** `review` records `SKIPPED` where an input was not
supplied, which is visible in the ruling and in the journal. A session that forgets to supply
volatility does not silently disable the volatility limit; it produces rulings that say the limit
never ran.

## Looking at it## Looking at it

```julia
using BayesTrade, Dates
payload = dashboard_payload(report; book = book, generated_at = now(UTC) + IST_OFFSET)
write_dashboard_page(payload, "dashboard.html")     # open it from the filesystem
```

One HTML file with the payload inside it. No build step, no server, no network: every pixel is
drawn into a canvas by hand rather than by a charting library, because a library here means
either a CDN request the page cannot make offline or a vendored megabyte in the repository.

It leads with calibration rather than with an equity curve. Anyone can plot an equity curve;
whether the intervals hold up is what says the curve means anything. Then the posterior band
against what actually happened, the coverage diagram, the pool weights over time, and every
decision with the gate that stopped it.

While something is still running, serve it instead:

```julia
using HTTP
server = serve_dashboard(() -> dashboard_payload(latest_report(); book = book,
                                                 generated_at = now(UTC) + IST_OFFSET);
                         refresh_seconds = 30)
close(server)
```

The producer is called per request, so the page shows the current state rather than the state
it had at startup. **It binds to the loopback interface and there is no keyword to change
that.** The page carries positions, limits and every decision the system took; putting that on
whatever network the machine is attached to, with no authentication, is not something a flag
should make easy.

`examples/dashboard.jl` does the whole thing, on Groww history when `GROWW_API_KEY` is set and
on synthetic history otherwise.

## Safety gates

Four independent barriers stand between running this and sending a real order. The first is the
strongest because it is not a check:

1. **No live broker exists.** `subtypes(Broker) == [PaperBroker]`. There is no implementation
   that can reach a venue. A test asserts this and will fail the moment one is added.
2. **`PaperTradingSession.broker` is typed `PaperBroker`**, so a live broker could not be
   installed there even if one existed.
3. **A live `KiteSession` requires `BAYESTRADE_ALLOW_LIVE_TRADING=true`** in addition to being
   asked for live mode.
4. **`load_settings()` refuses a half-set live configuration**, so one variable set by accident
   does nothing.

The Kite client can authenticate and read quotes. It cannot place an order: there is no
`place_order!` method that speaks to it.

The Groww client is narrower still. `build_request` refuses any path outside
`GROWW_READ_PATHS`, which holds two entries: mint a token, and read historical candles. Groww's
API has order, position and holding endpoints; none of them can be reached from here, and the
refusal happens when the request is built rather than when it is sent.

## Historical data

`GrowwSource` fetches real NSE history for backtesting. It needs `GROWW_API_KEY` in the
environment, plus `GROWW_API_SECRET` for an approval key or a TOTP code passed to
`authenticate!` for a TOTP key. Credentials are read at run time and belong nowhere else: not
in a file in this repository, not in a saved script, not in a message. A secret that has been
seen once should be regenerated rather than reused.

Tokens expire daily at 06:00 IST. `session.expiry` records the expiry Groww returned, and
`token_expired(session, moment)` answers whether it has passed, so a long run can mint a new
one instead of discovering the fact halfway through a fetch.

Three behaviours matter for correctness rather than convenience:

- A window wider than Groww serves in one request is split and stitched. Truncating it would
  leave a shorter series that still looks like a series.
- Candles arriving twice at chunk boundaries are stored once. Two bars at one timestamp is a
  repeated observation, and it moves every estimate that counts observations.
- The candle covering the current period is dropped. Its close is not a close, and a model
  fitted on it trains on a number that did not exist at that time.

Market data on Groww is a paid entitlement. Quotes, OHLC and historical candles all sit behind
the Trading API subscription, and a 403 on `/historical/candles` after a token was minted
successfully means the subscription is not active rather than that anything is misconfigured.
`translate_error` says so, because without the hint it reads as a bug in the client.

The transport is a weak dependency. `BayesTrade` resolves, precompiles, backtests and paper
trades with no HTTP library present, and `groww_transport` only exists once `using HTTP` has
been run; without it, both it and `connect_groww` raise and say what is missing rather than
failing as a `MethodError`.

**Do not `Pkg.add("HTTP")` inside the package project.** It promotes the weak dependency to a
real one and rewrites `Project.toml`, deleting the `[weakdeps]` and `[extensions]` blocks with
it. The examples carry their own environment for exactly this reason:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=examples examples/groww_history.jl RELIANCE
```

```julia
using BayesTrade, HTTP

source = connect_groww()                        # reads the environment, authenticates
bars = fetch_bars(source, "RELIANCE"; start = Date(2025, 1, 1), stop = Date(2026, 6, 30))
```

`connect_groww` has no keyword for a secret. An argument is a thing that ends up in a script, a
shell history and a stack trace, so the only way in is the environment. A TOTP code is accepted
as an argument because it is worthless thirty seconds later.

`examples/groww_history.jl` runs the whole path: authenticate, fetch, quality-check, replay.
