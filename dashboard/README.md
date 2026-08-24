# Dashboard

A single self-contained page fed by a JSON payload. No Julia type crosses that
line, so this can be rewritten in anything or deleted without touching a line of
code that decides a trade.

## One command

```sh
./deploy.sh           # replay, rebuild, publish
./deploy.sh --local   # replay and rebuild only
```

Three steps, each usable on its own:

| Step | What it does |
| --- | --- |
| `generate.jl` | replays the full pipeline and writes `payload.json` |
| `build.jl` | folds the payload into `template.html`, writes `index.html` |
| `vercel deploy` | publishes it |

The output is reproducible: running it twice on unchanged code produces a
byte-identical `index.html`. `build.jl` refuses a payload that is missing, empty
or not a JSON object, so a broken run fails loudly instead of publishing a page
with no numbers in it.

## Why a template and a payload rather than one file

`index.html` is around 850 kB because the data is baked in, and a file that size
is not something anyone can edit. `template.html` is 22 kB of page and
`payload.json` is the rest, and each is editable on its own. Only the built
artefact is large, and it is regenerated rather than hand-maintained.

## What it draws

Two scenarios, chosen so the page shows both halves of the system's behaviour:

- **Real edge** — returns carrying genuine one-bar autocorrelation. The system
  finds it, trades it, and pays every cost.
- **Pure noise** — regime-switching returns with no exploitable drift. The system
  declines every single bar.

Showing only the second makes the system look inert; showing only the first makes
it look like a backtest. Together they are the actual claim: it trades what is
there and refuses what is not.

Both markets come from generators, so the edge in the first was put there on
purpose and is far stronger than anything real. The page says so at the top,
because an interviewer will ask and the answer is better volunteered than
extracted.

The hero is the system's actual output rather than a price line: every bar's
predictive density, drawn as a ridge in time, return and density. Beneath it sit
calibration, model disagreement, the reducible share of each prediction, and
every refusal with its gate named — the numbers that say whether an equity curve
means anything.

## Not yet

No HTTP server on the Julia side, so the page carries a payload rather than
polling an endpoint. When a server exists the only change is where the payload
comes from.
