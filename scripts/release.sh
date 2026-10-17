#!/usr/bin/env bash
# release.sh — GitHub Release 自动发布 CLI（谷歌双语模板流程）
#
# 流程: 前置检查 → 版本号推断 → 自动生成双语 release notes 脚手架 → gh release create → 校验
#
# 用法:
#   bash scripts/release.sh --prepare [vX.Y.Z]   # 前置检查 + 版本推断 + 生成 notes 脚手架
#   bash scripts/release.sh --notes <file>       # 用准备好的 notes 创建 release（推荐）
#   bash scripts/release.sh --use-scaffold       # 以脚手架内容直接发布（未润色，少用）
#   附加: --draft / --prerelease / --allow-dirty / --dry-run / --title "主题短语"
#
# 润色脚手架时严格遵循: scripts/RELEASE_NOTES_TEMPLATE.md
# 提交提取规则见: scripts/extract_commits.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE=""
NOTES_FILE=""
TITLE_SUFFIX=""
VERSION_ARG=""
FORCE_DRAFT=false
PRERELEASE=false
ALLOW_DIRTY=false
DRY_RUN=false
USE_SCAFFOLD=false

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare) MODE="prepare" ;;
    --notes) MODE="publish"; NOTES_FILE="$2"; shift ;;
    --use-scaffold) MODE="publish"; USE_SCAFFOLD=true ;;
    --draft) FORCE_DRAFT=true ;;
    --prerelease) PRERELEASE=true ;;
    --allow-dirty) ALLOW_DIRTY=true ;;
    --dry-run) DRY_RUN=true ;;
    --title) TITLE_SUFFIX="$2"; shift ;;
    -h|--help) usage ;;
    v*) VERSION_ARG="$1" ;;
    *) echo "未知参数: $1（--help 查看用法）"; exit 1 ;;
  esac
  shift
done

[[ -n "$MODE" ]] || { echo "未指定模式：--prepare 或 --notes/--use-scaffold（--help）"; exit 1; }

info()  { echo "[release] $*"; }
die()   { echo "[release][错误] $*" >&2; exit 1; }

# ---------- 1. 前置检查 ----------
info "前置检查 (Preflight)..."
has_gh() { command -v gh >/dev/null 2>&1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "不在 git 仓库中"
! has_gh && die "gh CLI 未安装（sudo apt install gh / brew install gh）"
gh auth status >/dev/null 2>&1 || die "gh 未登录（gh auth login）"
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
[[ -n "$ORIGIN_URL" ]] || die "缺少 origin remote"
REMOTE_REPO="${ORIGIN_URL#*github.com[:/]}"; REMOTE_REPO="${REMOTE_REPO%.git}"
info "仓库 (Repo): $REMOTE_REPO"
UNTRACKED="$(git status --porcelain | head -5 || true)"
if [[ -n "${UNTRACKED//$' '/}" ]] && [[ "$ALLOW_DIRTY" != true ]]; then
  echo "$UNTRACKED"
  die "工作区有未提交改动（确认无碍可用 --allow-dirty 继续）"
fi
git fetch --tags --quiet 2>/dev/null || info "fetch --tags 失败（离线？继续本地 tag）"

PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
if [[ -n "$PREV_TAG" ]]; then
  RANGE="${PREV_TAG}..HEAD"
  info "上一个版本 (Prev tag): ${PREV_TAG}"
else
  RANGE="HEAD"
  info "仓库尚无 tag（首次发版）"
fi

# ---------- 2. 版本号 ----------
N_COMMITS="$(git log --no-merges --oneline ${RANGE} 2>/dev/null | wc -l | tr -d ' ')"
[[ "$N_COMMITS" -gt 0 ]] || die "自 $PREV_TAG 以来没有变更，不发布空版本"
info "自 ${PREV_TAG:-仓库起点} 以来的提交: ${N_COMMITS} 个"

PYPROJECT_VERSION="$(python3 - <<'PY'
import tomllib
with open('pyproject.toml','rb') as f:
    print(tomllib.load(f)['project']['version'])
PY
)"

is_breaking() { git log --no-merges -E --grep='!:' --grep='^BREAKING[ -]CHANGE' --pretty=format:'%s' ${RANGE} | grep -q .; }
has_feat()    { git log --no-merges --pretty=format:'%s' ${RANGE} | grep -qE '^(feat|新增|加)'; }

bump() {
  local v="$1" kind="$2"
  local a b c
  IFS='.' read -r a b c <<<"${v#v}"
  case "$kind" in
    major) echo "v$((a+1)).0.0" ;;
    minor) echo "v${a}.$((b+1)).0" ;;
    patch) echo "v${a}.${b}.$((c+1))" ;;
  esac
}

if [[ -n "$VERSION_ARG" ]]; then
  VERSION="${VERSION_ARG#v}"; VERSION="v${VERSION}"
  info "使用指定版本: $VERSION"
elif [[ -z "$PREV_TAG" ]]; then
  VERSION="v${PYPROJECT_VERSION}"
  info "首次发版，沿用 pyproject 版本: $VERSION"
elif is_breaking; then
  KIND=minor
  # 0.x 阶段破坏性变更按定义升级 minor；≥1.x 升 major
  [[ "${PYPROJECT_VERSION%%.*}" -ge 1 ]] && KIND=major
  VERSION="$(bump "$PREV_TAG" "$KIND")"
  info "检测到 BREAKING 变更 → $KIND 升级"
elif has_feat; then
  VERSION="$(bump "$PREV_TAG" minor)"
  info "检测到 feat 提交 → minor 升级"
else
  VERSION="$(bump "$PREV_TAG" patch)"
  info "仅 fix/perf/docs 等 → patch 升级"
fi

if [[ "v${PYPROJECT_VERSION}" != "$VERSION" ]]; then
  info "提醒: pyproject.toml 版本为 ${PYPROJECT_VERSION}，与发布版本 ${VERSION} 不一致，建议先同步再发布。"
fi

PROJ_TITLE="Channel-Estimation-Methods-Comparison-Framework"
DATE="$(date +%F)"
COMPARE_LINK="https://github.com/${REMOTE_REPO}/compare/${PREV_TAG:-$(git rev-list --max-parents=0 HEAD | head -c7)}...${VERSION}"

mkdir -p release-notes
SCAFFOLD="release-notes/${VERSION}.md"

# ---------- 3. notes ----------
if [[ "$MODE" == "prepare" ]]; then
  bash scripts/extract_commits.sh ${PREV_TAG:+"$PREV_TAG"} HEAD > "${SCAFFOLD%.md}-commits.txt" || true
  {
    echo "# ${PROJ_TITLE} ${VERSION} Release Notes"
    echo ""
    echo "> 发布日期 Release Date: ${DATE} ｜ Full Changelog: ${COMPARE_LINK}"
    echo ""
    echo "<!-- 发布前按 scripts/RELEASE_NOTES_TEMPLATE.md 润色："
    echo "     每条两行——第一行英文(Added/Improved/Fixed 开头)，两格缩进中文行，"
    echo "     条目末尾补 (\#N)；无内容的段落整段删除；Known Issues 保留写 None / 无 -->"
    echo ""
    if is_breaking; then echo "## ⚠️ BREAKING CHANGES ｜ 破坏性变更"; echo ""; fi
    echo "## 🎉 New Features ｜ 新功能"
    echo ""
    echo "## ⚡ Enhancements ｜ 功能优化"
    echo ""
    echo "## 🐛 Bug Fixes ｜ 问题修复"
    echo ""
    echo "## ❗ Known Issues ｜ 已知问题"
    echo "- None / 无"
    echo ""
    echo "<!-- 原始提交清单见 ${SCAFFOLD%.md}-commits.txt -->"
  } > "$SCAFFOLD"
  info "脚手架已生成: $SCAFFOLD（提交清单: ${SCAFFOLD%.md}-commits.txt）"
  info "下一步: 润色后执行 bash scripts/release.sh --notes $SCAFFOLD${DRY_RUN:+ --dry-run}${FORCE_DRAFT:+ --draft}"
  exit 0
fi

# publish 模式：notes 来源
if [[ -z "$NOTES_FILE" ]]; then
  NOTES_FILE="$SCAFFOLD"
  [[ "$USE_SCAFFOLD" == true ]] || die "--use-scaffold 需要显式开启；或先 --prepare 生成脚手架并润色后用 --notes 发布"
fi
[[ -f "$NOTES_FILE" ]] || die "notes 文件不存在: $NOTES_FILE"
[[ "$USE_SCAFFOLD" == true && ! -f "$SCAFFOLD" ]] && die "脚手架不存在，请先 --prepare"
[[ "$USE_SCAFFOLD" == true ]] && NOTES_FILE="$SCAFFOLD"
info "使用 notes: $NOTES_FILE"

# ---------- 4. 发布 ----------
BRANCH="$(git branch --show-current || echo main)"
FLAGS=()
if is_breaking || [[ "${VERSION}" == v1.0.0 ]] || [[ "$FORCE_DRAFT" == true ]]; then FLAGS+=(--draft); info "将创建草稿 release（重大/破坏性版本需人工确认转正）"
else FLAGS+=(--latest); fi
$PRERELEASE && FLAGS+=(--prerelease)
TITLE="$VERSION"; [[ -n "$TITLE_SUFFIX" ]] && TITLE="${VERSION} ｜ ${TITLE_SUFFIX}"

if [[ "$DRY_RUN" == true ]]; then
  info "(dry-run) 将执行:"
  echo "  gh release create $VERSION --target $BRANCH --title \"$TITLE\" --notes-file $NOTES_FILE ${FLAGS[*]:-}"
  exit 0
fi

info "创建 release: $VERSION (target: $BRANCH)"
gh release create "$VERSION" --target "$BRANCH" --title "$TITLE" \
  --notes-file "$NOTES_FILE" ${FLAGS[*]+"${FLAGS[@]}"}

# ---------- 5. 校验 ----------
gh release view "$VERSION" --json tagName,isDraft,isPrerelease,url \
  --jq '"tagName: \(.tagName) | draft: \(.isDraft) | prerelease: \(.isPrerelease)\nURL: \(.url)"'
info "发布完成 ✅"
