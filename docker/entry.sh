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
export RCLONE_CONFIG="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
export R2_RCLONE_PROVIDER="${R2_RCLONE_PROVIDER:-Other}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$(dirname "$RCLONE_CONFIG")"

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER="$R2_RCLONE_PROVIDER"
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"
export RCLONE_CONFIG_R2_REGION=auto
export RCLONE_CONFIG_R2_ACL=private

cat >"$RCLONE_CONFIG" <<EOF
[r2]
type = s3
provider = $R2_RCLONE_PROVIDER
access_key_id = $R2_ACCESS_ID
secret_access_key = $R2_ACCESS_KEY
endpoint = $R2_ENDPOINT
region = auto
acl = private
EOF
chmod 600 "$RCLONE_CONFIG"

exec "$@"
