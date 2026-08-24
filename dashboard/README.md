# Dashboard

A single self-contained page. It reads a JSON payload produced by
`dashboard_payload()` and draws it; no Julia type crosses that line, so this
can be rewritten in anything or thrown away without touching a line of code
that decides a trade.

## Regenerating the payload

```julia
using BayesTrade, Dates

report  = replay(factories, examples; config = ReplayConfig(warmup = 800))
payload = dashboard_payload(
    report;
    book = portfolio(broker; as_of = now_bar),
    sector = "energy",
    generated_at = now_bar,
)
write_dashboard(payload, "dashboard/payload.json")
```

`index.html` currently inlines a downsampled payload so the page opens with no
server at all. The full payload for a 1778-bar replay is about 1.5 MB, which is
fine over HTTP and wasteful inline, so the inlined copy carries every aggregate
and roughly 300 of the series points.

## What it shows, and why those things

The hero is 297 predictive distributions drawn as a ridge in time, return and
density. That is the system's actual output: it never emits a price target, it
emits a shape, and the shape is what the risk engine rules on.

Everything below the ridge is the uncomfortable half. Any dashboard can draw an
equity curve. This one leads with calibration, model disagreement, the share of
each prediction that is reducible uncertainty, and every refusal with its gate
named, because those are the numbers that say whether an equity curve means
anything at all.

The current replay declines on every single bar. That is the correct answer on
a generator with no exploitable drift, and the page says so rather than showing
an empty blotter.

## Deploying

The page is static, so any file host works. It currently sits on Vercel:

```sh
cd dashboard && vercel deploy --prod
```

`.vercel/` holds the project link and is ignored; deleting it means the next
deploy creates a new project rather than updating this one.

## Not yet

There is no HTTP server on the Julia side, so the page reads a file rather than
an endpoint. When one exists, the only change here is where the payload comes
from.
