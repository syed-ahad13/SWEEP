#!/usr/bin/env bash
# Reference copy of the "On-start Script" pasted into the Vast template.
# Paste the body (everything below the SNIP line) into the template's On start field.
# Runs as root at container start, before you connect. Output: /root/init.log
#
# Template Docker options must contain:
#   -e GH_TOKEN=github_pat_xxxxxxxx
# Optional (note: an API key bills Console credits, NOT your Claude subscription):
#   -e ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
#
# Keep the template PRIVATE. It contains a token.
#
# No `set -e`: this script is never watched while it runs, so a failing step must
# report and continue rather than abort the whole bootstrap in silence.
# ------------------------------------------------------------------ SNIP
exec >>/root/init.log 2>&1
echo
echo "=== onstart $(date -u +%FT%TZ) ==="

REPO=syed-ahad13/SWEEP
DEST=/workspace/SWEEP
BRC=/root/.bashrc

ok()   { echo "[ok]   $*"; }
warn() { echo "[warn] $*"; }
bad()  { echo "[FAIL] $*"; }
line() { grep -qF "$1" "$BRC" 2>/dev/null || printf '%s\n' "$1" >>"$BRC"; }

# --- 1. environment sshd would otherwise discard ---------------------------
# Docker ENV from the image (and -e vars) live in PID 1's environment. An sshd
# login shell rebuilds PATH from /etc/profile, so CUDA and the token must be
# written into .bashrc or they vanish the moment you connect.
line 'export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"'
line 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"'
line 'export CUDA_DEVICE_ORDER=PCI_BUS_ID'
line 'export CMAKE_CUDA_ARCHITECTURES=89'
export PATH="/usr/local/cuda/bin:/root/.local/bin:$PATH"

if [ -n "${GH_TOKEN:-}" ]; then
  line "export GH_TOKEN=$GH_TOKEN"
  ok "GH_TOKEN present"
else
  bad "GH_TOKEN missing from the template's Docker options - clone will fail"
fi
[ -z "${ANTHROPIC_API_KEY:-}" ] || line "export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"

# --- 2. minimum needed to reach the repo -----------------------------------
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do
  apt-get update -qq && break || { warn "apt update failed, retry $i"; sleep 5; }
done
apt-get install -y -qq --no-install-recommends git curl ca-certificates \
  && ok "base packages" || bad "base packages"

# --- 3. repo --------------------------------------------------------------
git config --global credential."https://github.com".helper \
  '!f() { test "$1" = get && printf "username=x-access-token\npassword=%s\n" "$GH_TOKEN"; }; f'
git config --global safe.directory '*'
git config --global user.name  "Ahad"
git config --global user.email "abdulahad17100@gmail.com"
mkdir -p /workspace

if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only --quiet && ok "repo updated" || warn "pull failed"
elif git clone --quiet "https://github.com/$REPO.git" "$DEST"; then
  ok "repo cloned"
else
  bad "clone failed - check the token's Contents:write permission on $REPO"
fi

# --- 4. dependencies, owned by the repo not the template -------------------
if [ -f "$DEST/scripts/deps.sh" ]; then
  bash "$DEST/scripts/deps.sh" && ok "deps.sh" || bad "deps.sh"
else
  warn "no scripts/deps.sh - installing fallback set"
  apt-get install -y -qq --no-install-recommends tmux build-essential
fi

# --- 5. claude code -------------------------------------------------------
[ -x /root/.local/bin/claude ] || curl -fsSL https://claude.ai/install.sh | bash
[ -x /root/.local/bin/claude ] && ok "claude code" || warn "claude code install failed"

# --- 6. make login useful: land in the repo, show the verdict --------------
if ! grep -qF 'sweep-login' "$BRC" 2>/dev/null; then
  cat >>"$BRC" <<'EOF'
# sweep-login
if [ -n "$PS1" ]; then
  cd /workspace/SWEEP 2>/dev/null
  [ -f /root/init.status ] && cat /root/init.status
fi
EOF
fi

# --- 7. host gate ---------------------------------------------------------
if [ -f "$DEST/scripts/pod_init.sh" ] && bash "$DEST/scripts/pod_init.sh"; then
  echo "GATE PASS - $(date -u +%FT%TZ)" >/root/init.status
else
  echo "GATE FAIL - destroy this instance (see /root/init.log)" >/root/init.status
fi
cat /root/init.status
echo "=== onstart done ==="
