#!/usr/bin/env bash
# Builds the Linux binaries and copies them with the cluster scripts to
# /scratch/$USER/vsi/code on the cluster.
#   cluster/push.sh            # host alias "explorer"
#   VSI_HOST=myhpc cluster/push.sh
#
# It needs an already-open ssh master connection and will not open one: every
# new login costs a Duo prompt, and repeated failed logins lock the account
# out of the login nodes. Open it yourself first with `ssh explorer true`.
set -euo pipefail
HOST=${VSI_HOST:-explorer}
if ! ssh -O check "$HOST" 2>/dev/null; then
  echo "no ssh master connection to $HOST; run: ssh $HOST true" >&2
  exit 1
fi
cd "$(dirname "$0")/.."
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o bin/linux-amd64/ ./cmd/...

remote_user=$(ssh "$HOST" 'echo $USER')
dest=/scratch/$remote_user/vsi/code
# tar over ssh: scp to this cluster drops connections intermittently. On macOS,
# COPYFILE_DISABLE and the two flags keep AppleDouble ._ files and extended
# attributes out of the archive.
tar_flags=()
if [ "$(uname)" = Darwin ]; then
  export COPYFILE_DISABLE=1
  tar_flags=(--no-mac-metadata --no-xattrs)
fi
tar "${tar_flags[@]}" -cf - cluster bin/linux-amd64 |
  ssh "$HOST" "mkdir -p $dest /scratch/$remote_user/vsi/logs && tar -C $dest -xf -"
echo "pushed to $HOST:$dest"
