#!/usr/bin/env bash
# Regenerate, rebuild and publish the dashboard.
#
#   ./deploy.sh          generate, build, deploy to production
#   ./deploy.sh --local  generate and build only, no deploy
#
# Every step is idempotent, so running it twice on unchanged code produces the
# same page.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> replaying the pipeline"
julia --project=.. generate.jl

echo "==> building the page"
julia build.jl

if [[ "${1:-}" == "--local" ]]; then
  echo "==> built at $(pwd)/index.html (not deployed)"
  exit 0
fi

command -v vercel >/dev/null || { echo "vercel CLI not found; page built but not deployed" >&2; exit 1; }
echo "==> deploying"
vercel deploy --prod --yes
