#!/usr/bin/env sh
set -eu

: "${HF_TOKEN:?HF_TOKEN is required}"

R2_ACCESS_ID="${R2_ACCESS_ID:-${r2_access_id:-}}"
R2_ACCESS_KEY="${R2_ACCESS_KEY:-${r2_access_key:-}}"
R2_ENDPOINT="${R2_ENDPOINT:-${r2_endpoint:-}}"

: "${R2_ACCESS_ID:?R2_ACCESS_ID is required}"
: "${R2_ACCESS_KEY:?R2_ACCESS_KEY is required}"
: "${R2_ENDPOINT:?R2_ENDPOINT is required}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"
export RCLONE_CONFIG_R2_REGION=auto
export RCLONE_CONFIG_R2_ACL=private

exec "$@"
