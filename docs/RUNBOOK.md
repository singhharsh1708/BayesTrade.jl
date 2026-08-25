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

Kill the process. There is nothing to unwind:

- no live orders exist, because no live broker exists
- paper positions live in memory and are discarded
- the journal holds everything up to the last completed bar

If a live broker is ever added, this section stops being adequate and must be rewritten before
that broker is merged.

## Feed failure

The session halts itself. A tick arriving after more than `max_silence` of quiet makes
`check_health` fail on `:feed`, and the bar is recorded as `halted` with the gap named rather
than traded.

Nothing is required of the operator except to notice. To confirm:

```sh
grep '"event":"halted"' journal.jsonl | tail
```

Do not restart into a live feed and assume the gap is closed. The models will absorb bars again
as soon as the feed is healthy, and the missed bars are simply missing rather than filled in.

## Broker failure

Not applicable in paper mode: the broker is in-process and cannot time out.

When a live broker exists, the required behaviour is: stop sending new orders, reconcile the
positions the venue reports against the positions the session believes it holds, and refuse to
trade until they agree. That reconciliation does not exist yet and is a blocker for Phase 15.

## Database failure

There is no database. State is the journal file plus in-memory session state.

If the journal cannot be written, `record!` remembers the failure and `check_health` fails on
`:journal`, which halts trading. A decision that cannot be written down cannot be explained
afterwards, and an unexplainable trade is one that should not have happened.

## Model failure

`check_health` fails on `:models` or `:model_state` when any model is unfitted, and the session
will not predict. There is no path that trades on partial model state: `predict` on an unfitted
model raises `NotFittedError` rather than returning a prior-only guess.

## Restart

Feed the same bars and the session reaches the same state; there is a test asserting it. The
journal is append-only, so a restart appends rather than truncating.

What a restart does **not** do is recover the in-memory position book. In paper mode that is
acceptable because the positions are notional. It is not acceptable for live trading, and
rebuilding position state from the journal is a blocker for Phase 15.

See `docs/PRE_LIVE_AUDIT.md` for the go/no-go verdict, the defects the audit found, and the
blockers still open.

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

The transport is a weak dependency. `BayesTrade` resolves, precompiles, backtests and paper
trades with no HTTP library present, and `groww_transport` only exists once `using HTTP` has
been run; without it, both it and `connect_groww` raise and say what is missing rather than
failing as a `MethodError`. Add HTTP to your own environment:

```julia
using Pkg; Pkg.add("HTTP")
using BayesTrade, HTTP

source = connect_groww()                        # reads the environment, authenticates
bars = fetch_bars(source, "RELIANCE"; start = Date(2025, 1, 1), stop = Date(2026, 6, 30))
```

`connect_groww` has no keyword for a secret. An argument is a thing that ends up in a script, a
shell history and a stack trace, so the only way in is the environment. A TOTP code is accepted
as an argument because it is worthless thirty seconds later.

`examples/groww_history.jl` runs the whole path: authenticate, fetch, quality-check, replay.
