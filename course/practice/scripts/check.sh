#!/usr/bin/env bash
# scripts/check.sh — 跑 course/practice 下所有 lab 的 verify
# 用法：
#   bash scripts/check.sh             # 真跑（需要 gcc）
#   bash scripts/check.sh --dry-run   # 只打印每个 lab 会跑什么

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS_DIR="$SCRIPT_DIR/../labs"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

total=0
pass=0
fail=0

for lab_dir in "$LABS_DIR"/lab*; do
  [ -d "$lab_dir" ] || continue
  name=$(basename "$lab_dir")
  makefile="$lab_dir/Makefile"
  if [ ! -f "$makefile" ]; then
    echo "[SKIP] $name (no Makefile)"
    continue
  fi
  total=$((total + 1))
  echo "=== $name ==="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  would run: cd $lab_dir && make clean && make test"
    continue
  fi
  if (cd "$lab_dir" && make clean >/dev/null 2>&1 && make test); then
    pass=$((pass + 1))
    echo "  [PASS]"
  else
    fail=$((fail + 1))
    echo "  [FAIL]"
  fi
done

echo
echo "================================"
echo "Lab summary: $pass passed, $fail failed, $total total"
[ "$DRY_RUN" = "1" ] && echo "(dry-run: nothing was actually run)"
echo "================================"
