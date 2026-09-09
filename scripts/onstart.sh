#!/usr/bin/env bash
# Reference copy of the "On-start Script" pasted into the Vast template.
# Paste the body (everything below the SNIP line) into the template's On start field.
# It runs as root at container start, before you connect. Output goes to /root/init.log.
#
# Requires one template environment variable, set in the template's Docker options:
#   -e GH_TOKEN=github_pat_xxxxxxxx
# Optional, makes Claude Code zero-touch instead of a 10-second login:
#   -e ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
#
# Keep the template PRIVATE. It contains a token.
# ------------------------------------------------------------------ SNIP
set -eu
exec >>/root/init.log 2>&1
echo "=== onstart $(date -u +%FT%TZ) ==="

REPO=syed-ahad13/SWEEP
DEST=/workspace/SWEEP

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  git curl ca-certificates tmux build-essential

# Make the token available to interactive SSH shells. Docker -e vars land in PID 1's
# environment; an sshd-spawned login shell does not inherit them.
grep -q GH_TOKEN /root/.bashrc 2>/dev/null || printf 'export GH_TOKEN=%s\n' "$GH_TOKEN" >>/root/.bashrc
[ -z "${ANTHROPIC_API_KEY:-}" ] || grep -q ANTHROPIC_API_KEY /root/.bashrc 2>/dev/null \
  || printf 'export ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" >>/root/.bashrc

# Credential helper reads $GH_TOKEN at each git operation, so the token never enters
# a remote URL, .git/config, or argv.
git config --global credential."https://github.com".helper \
  '!f() { test "$1" = get && printf "username=x-access-token\npassword=%s\n" "$GH_TOKEN"; }; f'
git config --global --add safe.directory "$DEST"

mkdir -p /workspace
[ -d "$DEST/.git" ] || git clone --quiet "https://github.com/$REPO.git" "$DEST"

command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash
grep -q '.local/bin' /root/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >>/root/.bashrc

bash "$DEST/scripts/pod_init.sh" || echo "GATE FAILED - destroy this instance"
echo "=== onstart done ==="
