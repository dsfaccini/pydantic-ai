#!/usr/bin/env bash
# Run a command under a machine-wide concurrency limit for pyright typechecks.
#
# pyright spawns a Node process that uses a full CPU core and multi-GB RAM. Running
# several at once (e.g. one agent/worktree per checkout) can overwhelm a machine. This
# wrapper makes typecheck runs share a counting semaphore across ALL worktrees/clones
# for the current user, blocking until a slot is free, then runs the given command.
#
# Limit: PYRIGHT_CONCURRENCY (default 5). Set 0 or non-numeric to disable (run uncapped).
# The limit is intentionally generous by default so it's a no-op in CI and for anyone
# who doesn't set it; set PYRIGHT_CONCURRENCY=1 to fully serialize on a small machine.
#
# Usage: scripts/pyright-with-limit.sh <command...>
#   e.g. scripts/pyright-with-limit.sh uv run pyright
set -uo pipefail

LIMIT="${PYRIGHT_CONCURRENCY:-5}"

# Disabled / uncapped: just run the command.
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -le 0 ]; then
  exec "$@"
fi

SEM_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/pyright-semaphore"
mkdir -p "$SEM_DIR"

MY_SLOT=""
release() { [ -n "$MY_SLOT" ] && rm -rf "$MY_SLOT" 2>/dev/null; }
trap release EXIT INT TERM

acquire() {
  local announced="" i slot pid
  while true; do
    # Reclaim slots whose holder process has died (crash-safe).
    for slot in "$SEM_DIR"/slot.*; do
      [ -d "$slot" ] || continue
      pid="$(cat "$slot/pid" 2>/dev/null || true)"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        rm -rf "$slot" 2>/dev/null || true
      fi
    done
    # Try to claim a free slot (mkdir is atomic).
    for ((i = 1; i <= LIMIT; i++)); do
      slot="$SEM_DIR/slot.$i"
      if mkdir "$slot" 2>/dev/null; then
        echo "$$" >"$slot/pid"
        MY_SLOT="$slot"
        return 0
      fi
    done
    if [ -z "$announced" ]; then
      echo "pyright: all $LIMIT typecheck slot(s) busy, waiting for one to free up..." >&2
      announced=1
    fi
    sleep 1
  done
}

acquire
"$@"
exit $?
