# Skills 目录 —— 方法论资产

这里放的是**给 Agent 看的方法论文档**，由 Skill Hub（M7）加载。
改这里的 markdown 就能改变 Agent 的行为，**不需要改代码、不需要发版**。

---

## 一个 Skill 长什么样

一个 `.md` 文件 = 一个 Skill。YAML frontmatter 定元数据，正文写方法论。

```markdown
---
name: douyin-viral-structure
description: 拆解抖音爆款的结构公式，写短视频脚本或做选题时用
scope: content_type
applies_to: [短视频]
stage: [策划, 脚本, 分镜]
priority: 50
version: 1.0.0
owner: 运营-张三
status: active
---

# 正文从这里开始

写具体的方法论、SOP、检查清单、正反例……
```

---

## 字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✅ | 唯一标识，kebab-case，和文件名保持一致 |
| `description` | ✅ | **最重要的字段**，见下方专门说明 |
| `scope` | ✅ | `global` / `content_type` / `pipeline_stage` / `account` |
| `applies_to` | | 适用的内容形态：`短视频` `图文` `文案` `图像` |
| `stage` | | 适用的流水线节点：`选题` `策划` `脚本` `分镜` `正文` `配图` `排版` `审核` |
| `priority` | | 0–100，默认 50。冲突时数字大的优先 |
| `version` | ✅ | 语义化版本，改内容就升 |
| `owner` | ✅ | 谁维护，出问题找谁 |
| `status` | ✅ | `active`（生效）/ `draft`（不加载）/ `placeholder`（占位，不加载） |

---

## ⚠️ `description` 怎么写 —— 这是唯一真正影响效果的字段

Skill Hub 采用**两级披露**：

- **常驻上下文的只有 `description`**（约 50 token/个）
- **正文只在被选中时才加载**（2–5K token/个）

也就是说，**Agent 是靠 `description` 来决定要不要读你这篇文档的**。写不好，文档写得再精彩也永远不会被翻开。

**写法：说清楚「什么情况下该用它」，而不是「它是什么」。**

| ❌ 不好 | ✅ 好 |
|---|---|
| 抖音方法论 | 拆解抖音爆款的结构公式，写短视频脚本或做选题时用 |
| 文案规范 | 小红书文案的语气与排版规范，写小红书正文或标题时必读 |
| 合规检查 | 平台违禁词与法务红线，任何内容发布前必须检查 |

一句话，20–40 字，包含**触发场景**和**动作**。

---

## 命名与组织

- 文件名 = `name` + `.md`，kebab-case，用英文（便于引用），中文写在正文里
- 平铺存放，不建子目录 —— Skill Hub 靠 frontmatter 的 `scope`/`applies_to`/`stage` 做筛选，不靠目录结构
- 以 `_` 开头的文件不会被加载（如 `_TEMPLATE.md`）

---

## 加载规则（两级漏斗）

```
全部 skill
   ↓ ① 规则预筛：status=active，且 scope/applies_to/stage 匹配当前节点
候选集
   ↓ ② 模型判断：读 description，选出真正需要的
激活集（建议同时 ≤ 3 个）
   ↓ 加载正文进上下文
```

**规则先筛、模型再选。** 当前节点是「写小红书文案」，`applies_to: [短视频]` 的 skill 在进入模型视野前就被过滤掉了，不消耗任何 token。

---

## 冲突怎么办

两个 Skill 给出矛盾指导时（比如通用规范说「多用短句」，某账号 skill 说「保持长句书面感」），
按 **scope 特异性 → priority** 排序：

```
account  >  content_type  >  pipeline_stage  >  global
```

Hub 会在注入上下文时**显式标注哪条优先**，不让模型自己猜。

---

## 预算约束

Skill Hub 和 MCP Hub 共用一个 `capability_budget`（见 `config/capability_budget.yaml`）。

- 元数据目录常驻：约 50 token × skill 数量
- 激活的正文：建议同时不超过 3 个、合计 15K token

**正文写太长会挤掉别的能力。** 单篇建议控制在 2–5K token（约 1500–3500 中文字）。写不下就拆成多个 skill，用 `stage` 区分。

---

## 新增一个 Skill

1. 复制 `_TEMPLATE.md`
2. 改文件名为 `你的-skill-名.md`
3. 填 frontmatter，**重点打磨 `description`**
4. 写正文
5. 把 `status` 改成 `active`

热加载，存盘即生效（下一轮对话 / 下一个节点就用新版）。

每个版本按内容哈希记在 `workspace/skills_history/`：
`agent skill history <name>` 看记录，`agent skill rollback <name>` 一键退回上一版，
`agent skill check` 体检（解析失败、正文超长、没写 owner）。改坏了退得回，所以放心写。

---

## 当前占位清单

以下是已建好槽位、**等待填充正文**的 skill（`status: placeholder`，Hub 不会加载）：

| 文件 | 覆盖 |
|---|---|
| `douyin-viral-structure.md` | 短视频 · 爆款结构 |
| `xiaohongshu-copy-tone.md` | 图文/短视频 · 小红书调性 |
| `wechat-longform-structure.md` | 图文 · 公众号长文 |
| `poster-prompt-handbook.md` | 图像 · 海报 prompt |
| `compliance-redlines.md` | 全局 · 合规红线 |
