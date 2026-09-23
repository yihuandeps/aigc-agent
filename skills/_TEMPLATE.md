---
name: your-skill-name
description: 一句话说清「什么情况下该用它」，20-40字。这是唯一影响是否被选中的字段，重点打磨。
scope: content_type          # global | content_type | pipeline_stage | account
applies_to: []               # 短视频 | 图文 | 文案 | 图像
stage: []                    # 选题 | 策划 | 脚本 | 分镜 | 正文 | 配图 | 排版 | 审核
priority: 50                 # 0-100，冲突时数字大的优先
version: 0.1.0
owner: TODO
status: draft                # active 才会被加载
---

# <Skill 标题>

> 建议正文控制在 2–5K token（约 1500–3500 中文字）。写不下就拆成多个 skill。

## 什么时候用

<描述适用场景。和 frontmatter 的 description 呼应，但可以展开。>

## 核心方法

<具体的方法论、公式、结构模板。越具体越好，避免"要写得有吸引力"这类无法执行的表述。>

## 检查清单

- [ ] <可勾选的、二元判断的检查项>
- [ ] <每一条都应该能明确回答"是"或"否">

## 正例

<给 1–2 个好的例子，并说明为什么好。>

## 反例

<给 1–2 个差的例子，并说明问题出在哪。反例往往比正例更有用。>

## 禁止事项

<绝对不能做的事。这部分会被 Memory Agent 优先提取进 Brief 的 must_not 区。>
