#!/usr/bin/env sh
set -eu

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${r2_access_id:?r2_access_id is required}"
: "${r2_access_key:?r2_access_key is required}"
: "${r2_endpoint:?r2_endpoint is required}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

rclone config create r2 s3 \
  provider Cloudflare \
  access_key_id "$r2_access_id" \
  secret_access_key "$r2_access_key" \
  endpoint "$r2_endpoint" \
  region auto \
  acl private \

exec "$@"