#!/usr/bin/env bash
# scripts/bootstrap-practice.sh — Chapter 0 进入实践框架的统一入口
# 用法：
#   bash scripts/bootstrap-practice.sh
#   bash scripts/bootstrap-practice.sh --lab lab03-step2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COURSE_DIR="$(cd "$ROOT_DIR/.." && pwd)"
LABS_DIR="$ROOT_DIR/labs"
TARGET_LAB="lab01-step0"

if [[ "${1:-}" == "--lab" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[ERROR] --lab 需要一个 lab 名称，例如 --lab lab03-step2" >&2
    exit 1
  fi
  TARGET_LAB="$2"
fi

LAB_DIR="$LABS_DIR/$TARGET_LAB"

if [[ ! -d "$LAB_DIR" ]]; then
  echo "[ERROR] lab 不存在: $LAB_DIR" >&2
  exit 1
fi

echo "========================================"
echo "miniLLM practice bootstrap"
echo "========================================"
echo "讲义目录: $COURSE_DIR"
echo "实践框架: $ROOT_DIR"
echo "目标 lab : $TARGET_LAB"
echo

for cmd in gcc make; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] 缺少命令: $cmd" >&2
    echo "请先安装最小 C 工具链，再重新运行本脚本。" >&2
    exit 1
  fi
done

echo "[1/3] 工具链就绪"
gcc --version | head -n 1
make --version | head -n 1
echo

echo "[2/3] 进入实践目录"
echo "cd $LAB_DIR"
echo

echo "[3/3] 运行 smoke test"
set +e
SMOKE_OUTPUT="$(
  cd "$LAB_DIR"
  make clean >/dev/null
  make test 2>&1
)"
SMOKE_STATUS=$?
set -e

printf '%s\n' "$SMOKE_OUTPUT"
echo

if [[ $SMOKE_STATUS -ne 0 ]]; then
  if [[ "$SMOKE_OUTPUT" != *"[TEST"* && "$SMOKE_OUTPUT" != *"test(s) FAILED."* && "$SMOKE_OUTPUT" != *"tests passed"* ]]; then
    echo "[ERROR] smoke test 没有进入验证阶段，请先修复编译或链接错误。" >&2
    exit "$SMOKE_STATUS"
  fi
fi

if [[ "$TARGET_LAB" == "lab01-step0" ]]; then
  cat <<'EOF'
预期说明：
- Chapter 0 的默认基线不是全 PASS
- 当前脚手架应表现为 3 个 FAIL + 1 个 PASS
- 这说明你已经进入学员实践骨架，而不是完整参考实现

下一步：
1. 返回 course/00-orientation.md 收尾
2. 阅读 course/chapters/ch01-step0-tensor.md
3. 打开 labs/lab01-step0/TASK.md 和 framework/student.c
EOF
else
  cat <<EOF
已完成 $TARGET_LAB 的编译与验证入口检查。
下一步请回到对应 chapter，再进入该 lab 的 TASK.md 和 framework/student.c。
EOF
fi
