#!/usr/bin/env bash
# extract_commits.sh — 从上一个 tag（或指定范围）提取 conventional commits，按类型分组输出。
#
# 用法:
#   bash scripts/extract_commits.sh                     # 自动: 最新可达 tag..HEAD；无 tag 则全部历史
#   bash scripts/extract_commits.sh <from_tag> [to_ref] # 例: extract_commits.sh v0.2.0 main
#
# 输出: 按类型分组的提交（hash + 日期 + subject），末尾附 BREAKING 提示与统计。
# 产物只供撰写 release notes 参考，行文要求见 scripts/RELEASE_NOTES_TEMPLATE.md。

set -euo pipefail

FROM="${1:-}"
TO="${2:-HEAD}"

if [[ -z "$FROM" ]]; then
  if git describe --tags --abbrev=0 >/dev/null 2>&1; then
    FROM="$(git describe --tags --abbrev=0)"
  fi
fi

if [[ -n "$FROM" ]]; then
  RANGE="${FROM}..${TO}"
  echo "RANGE: ${RANGE}"
else
  RANGE="HEAD"
  echo "RANGE: 全部历史（未找到任何 tag）"
fi
echo "=========================================="

# 主列表：hash / 日期 / subject，按 conventional commit 类型分组
git log --no-merges --date=short --pretty=format:'%h%x09%ad%x09%s' ${RANGE} | awk -F'\t' '
function cls(s) {
  if (s ~ /^[Ff]eat/)     return "FEAT"
  if (s ~ /^[Ff]ix/)      return "FIX"
  if (s ~ /^[Pp]erf/)     return "PERF"
  if (s ~ /^[Rr]efactor/) return "REFACTOR"
  if (s ~ /^[Dd]ocs?/)    return "DOCS"
  if (s ~ /^[Ss]tyle/)    return "STYLE"
  if (s ~ /^[Tt]est/)     return "TEST"
  if (s ~ /^[Bb]uild/)    return "BUILD"
  if (s ~ /^[Cc][Ii]:/)   return "CI"
  if (s ~ /^[Rr]evert/)   return "REVERT"
  if (s ~ /^[Cc]hore/)    return "CHORE"
  return "OTHER"
}
{
  t = cls($3)
  n[t]++
  item[t] = item[t] sprintf("%s  %s  %s\n", $1, $2, $3)
}
END {
  split("FEAT FIX PERF REFACTOR DOCS STYLE TEST BUILD CI REVERT CHORE OTHER", ord, " ")
  for (i = 1; i <= 12; i++) {
    t = ord[i]
    if (t in item) {
      printf "\n## %s (%d)\n", t, n[t]
      printf "%s", item[t]
    }
  }
}'

# BREAKING 提示：footer 声明或 subject 带 !:
echo ""
echo "## BREAKING CHANGES（需人工逐条确认）"
BREAKING="$(git log --no-merges -E --grep='^BREAKING[ -]CHANGE' --grep='!:' --pretty=format:'%h %s' ${RANGE} || true)"
if [[ -n "$BREAKING" ]]; then
  echo "$BREAKING"
  echo "提示: 用 git log --format=%B -1 <hash> 查看完整 BREAKING CHANGE 描述"
else
  echo "(none)"
fi

# 统计摘要
TOTAL="$(git log --no-merges --oneline ${RANGE} | wc -l | tr -d ' ')"
echo ""
echo "TOTAL: ${TOTAL} 个提交（不含 merge）"
