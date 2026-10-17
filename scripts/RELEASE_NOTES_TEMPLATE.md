# 双语 Release Notes 模板（Google 风格）

模板风格参照 Google 产品发布说明（Android Studio / Firebase / Google Cloud Release Notes）：
分类固定、动词开头、面向用户收益、每条独立可读。在此之上做中英双版。

## 段落顺序（固定）

1. `BREAKING CHANGES ｜ 破坏性变更` —— 仅在有破坏性变更时出现，永远置顶
2. `New Features ｜ 新功能`（feat）
3. `Enhancements ｜ 功能优化`（feat 的改进型、refactor 带来可感知收益的）
4. `Performance ｜ 性能提升`（perf）
5. `Bug Fixes ｜ 问题修复`（fix）
6. `Documentation ｜ 文档`（docs）
7. `Deprecations ｜ 弃用说明`（deprecate 类提交或 issue）
8. `Known Issues ｜ 已知问题`
9. `Upgrade Guide ｜ 升级指南` —— 有破坏性变更或迁移步骤时出现
10. `Acknowledgements ｜ 致谢` —— 有外部贡献者时出现

**省略规则**：本次发布没有内容的段落整段删除，不要留 "(none)" 占位；唯一例外是
Known Issues——它必须保留，没内容时写 "None / 无"（让用户知道你确认过没有已知问题）。

## 写法规则

- **交错双语（默认，Variant A）**：每条变更两个物理行——第一行英文，第二行以两空格
  缩进开头写中文，markdown 渲染后同属一个列表项。
- **每条必须带编号**：PR 或 issue 编号（#N），从提交记录 / `gh pr list` 中获取，不要编造。
- **动词开头、用户视角**：英文用 Added / Improved / Fixed / Removed；中文对应
  新增 / 优化 / 修复 / 移除。描述效果与收益，不描述实现过程。
  量化优先："启动时间缩短约 40%" 优于 "优化了启动"。
- **一条一个完整意思**：同一 PR 的多个 commit 合并成一条；不相关的内容不合并。
- **中文规范**：技术名词、命令、路径、版本号保留英文原样（如 `--config`、API、YAML）；
  其余用中文，标点用中文全角。
- **标题与链接**：一级标题 `# <项目名> vX.Y.Z Release Notes`；开头引用块放发布日期
  和 compare 链接：`https://github.com/<owner>/<repo>/compare/<prevTag>...<newTag>`

## Variant A：交错双语（默认）

```markdown
# <项目名> vX.Y.Z Release Notes

> 发布日期 Release Date: YYYY-MM-DD ｜ Full Changelog: https://github.com/<owner>/<repo>/compare/vA.B.C...vX.Y.Z

## ⚠️ BREAKING CHANGES ｜ 破坏性变更
- The `--config` flag now expects a YAML file instead of JSON.
  `--config` 参数现在接受 YAML 文件，不再支持 JSON 格式（#88）

## 🎉 New Features ｜ 新功能
- Added `supercli doctor` to diagnose common environment issues.
  新增 `supercli doctor` 命令，一键诊断常见环境问题（#75）
- Added multi-language UI support (English, 简体中文).
  新增多语言界面支持（English、简体中文）（#91）

## ⚡ Enhancements ｜ 功能优化
- Lazy-loading plugins reduced startup time by ~40%.
  插件改为懒加载，启动时间缩短约 40%（#81）

## 🐛 Bug Fixes ｜ 问题修复
- Fixed a crash when the config file is read-only.
  修复配置文件为只读时的崩溃问题（#63）

## ❗ Known Issues ｜ 已知问题
- `--watch` mode may miss changes on NFS mounts.
  `--watch` 模式在 NFS 挂载目录上可能漏检文件变更。

## 📦 Upgrade Guide ｜ 升级指南
- Convert existing JSON config with `supercli migrate --from-json`.
  使用 `supercli migrate --from-json` 迁移现有 JSON 配置。

## 🙏 Acknowledgements ｜ 致谢
Thanks @alice for the first contribution! 感谢 @alice 的首个贡献！
```

## Variant B：整段两版（用户要求"独立中英文版"时）

结构同上，但同一份内容先完整英文版、后完整中文版，段内标题只用单语：

```markdown
# <项目名> vX.Y.Z Release Notes

> 发布日期 Release Date: YYYY-MM-DD ｜ Full Changelog: <compare 链接>

---

## English

### ⚠️ BREAKING CHANGES
- The `--config` flag now expects a YAML file instead of JSON. (#88)

### 🎉 New Features
- Added `supercli doctor` to diagnose common environment issues. (#75)

### ❗ Known Issues
- `--watch` mode may miss changes on NFS mounts.

### 📦 Upgrade Guide
- Convert existing JSON config with `supercli migrate --from-json`.

---

## 中文

### ⚠️ 破坏性变更
- `--config` 参数现在接受 YAML 文件，不再支持 JSON 格式（#88）

### 🎉 新功能
- 新增 `supercli doctor` 命令，一键诊断常见环境问题（#75）

### ❗ 已知问题
- `--watch` 模式在 NFS 挂载目录上可能漏检文件变更。

### 📦 升级指南
- 使用 `supercli migrate --from-json` 迁移现有 JSON 配置。
```

## 从提交到条目的翻译对照

| 提交类型 | 归入段落 | 英文开头 | 中文开头 |
|---|---|---|---|
| `feat:`（新能力） | New Features | Added … | 新增… |
| `feat:`（改进现有） / `refactor:`（有感知收益） | Enhancements | Improved … / Changed … | 优化… / 调整… |
| `perf:` | Performance | Improved performance of … | 提升了…的性能 |
| `fix:` | Bug Fixes | Fixed … | 修复… |
| `docs:` | Documentation | Updated documentation for … | 更新了…的文档 |
| `revert:` / 移除功能 | Deprecations 或 BREAKING | Deprecated … / Removed … | 弃用… / 移除… |
| `test:` `build:` `ci:` `chore:` | 一般不写进 notes | — | — |

test/build/ci/chore 属于开发过程变更，除非影响用户（如 CI 改变产物名、依赖要求变化），
否则不进 release notes。

## 发布标题

`gh release create` 的 `--title` 默认用 `vX.Y.Z`；主题鲜明的版本可写成
`vX.Y.Z ｜ 主题短语`（例：`v2.0.0 ｜ 插件体系重构`），中英文标题保持简短。
