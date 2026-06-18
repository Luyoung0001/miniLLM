#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${1:-minillm-course}"
PRODUCTION_BRANCH="${2:-main}"

if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "CLOUDFLARE_API_TOKEN is required" >&2
  exit 1
fi

if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "CLOUDFLARE_ACCOUNT_ID is required" >&2
  exit 1
fi

npx wrangler pages project create "${PROJECT_NAME}" \
  --production-branch "${PRODUCTION_BRANCH}"
