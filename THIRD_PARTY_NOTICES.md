# 第三方内容声明（Third-Party Notices）

本仓库自己写的代码与文档按 [MIT 许可证](LICENSE) 开源。
下面列出的目录和文件来自第三方，**各自按原许可使用（或由原作者保留权利），不在本仓库 MIT 许可的范围内**。
如果你是权利人，希望修改署名或移除内容，请开一个 issue，会尽快处理。

| 来源 | 许可 | 用在本仓库哪里 |
|---|---|---|
| [0xsline/short-drama](https://github.com/0xsline/short-drama) | MIT | `skills/drama-script/` |
| [LearnPrompt/awesome-seedance](https://github.com/LearnPrompt/awesome-seedance) | 代码 MIT · 策划内容 CC BY 4.0 · 案例原文归原作者 | `skills/seedance-prompting/`、`scripts/build_seedance_skill.py` 生成的参考文档、`config/recipes/ugc-vlog.yaml`、`config/recipes/product-ad.yaml` |
| [procmeans/rainwell-douyin-viral-analyzer](https://github.com/procmeans/rainwell-douyin-viral-analyzer) | **上游未声明许可证** | `src/aigc_agent/domain/functions/vendor/douyin/`、`src/aigc_agent/domain/functions/vendor/assets/report.css`、`skills/douyin-viral-analyzer.md`、`skills/data-fetching.md` |
| [yaohaoliang141-max/ai-character-passport](https://github.com/yaohaoliang141-max/ai-character-passport) | 仅借鉴思路，未复制代码 | `src/aigc_agent/domain/storyboard/` |

---

## 1. 0xsline/short-drama（MIT）

- 上游：https://github.com/0xsline/short-drama
- 本仓库中的位置：`skills/drama-script/`
  - `references/` 下 8 篇创作方法参考（题材、钩子、开篇、付费卡点、节奏曲线、爽点矩阵、反派设计、合规清单）为上游原文；
  - `SKILL.md` 在上游方法总纲的基础上，加入了本 Agent 的存稿 / 人审 / 工程链对接说明。
- 许可：MIT，Copyright (c) 2025 0xsline，全文见文末。

## 2. LearnPrompt/awesome-seedance（代码 MIT / 策划内容 CC BY 4.0）

- 上游：https://github.com/LearnPrompt/awesome-seedance （commit `9927d9b`，2026-09-23 拉取）
- 本仓库中的位置：
  - `skills/seedance-prompting/data/`：上游的模板与案例数据（经过裁剪，来源记录在同目录的 `SOURCE.md`）；
  - `skills/seedance-prompting/references/`、`skills/seedance-prompting/SKILL.md`：由 `scripts/build_seedance_skill.py` 根据上面的数据生成的中文参考；
  - `config/recipes/ugc-vlog.yaml`、`config/recipes/product-ad.yaml`：从上游模板派生的短视频配方。
- 许可：
  - **代码**（生成脚本等）：MIT，Copyright (c) 2026 LearnPrompt，全文见文末；
  - **策划内容**（模板提炼、分类、统计、摘要）：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，署名 *awesome-seedance / goodcase.ai*；
  - **案例中的提示词原文与媒体**：版权归各自的原作者（见每条案例的 `sourceUrl` / `creator`）。上游以学习和记录为目的引用这些内容，本仓库沿用同样的方式，不对它们另行授权。
- 本仓库所做的修改（按 CC BY 4.0 要求注明）：每个模板只保留热度前 5 的案例，提示词原文截断到 1800 字；与本仓库自己的模板合并；翻译、整理成中文参考文档。

## 3. procmeans/rainwell-douyin-viral-analyzer（上游未声明许可证）

- 上游：https://github.com/procmeans/rainwell-douyin-viral-analyzer
- 本仓库中的位置：
  - `src/aigc_agent/domain/functions/vendor/douyin/dy_fetch.py`：上游脚本，修过一处接口名，记录在同目录的 `PATCHES.md`；
  - `src/aigc_agent/domain/functions/vendor/douyin/render_pdf.py`：上游脚本，未修改；
  - `src/aigc_agent/domain/functions/vendor/assets/report.css`：上游 PDF 报告样式，未修改（放在渲染脚本查找的位置）；
  - `skills/douyin-viral-analyzer.md`：在上游 `SKILL.md` 基础上加了本 Agent 的 frontmatter 和工具对接说明；
  - `skills/data-fetching.md`：在上游 `references/data-fetching.md` 基础上修改（当前是 `draft` 状态，Agent 不会加载）。
- 许可：**上游仓库没有声明开源许可证**。这些文件的著作权归原作者所有，**不适用本仓库的 MIT 许可**，这里只注明出处。
  如果你要再分发或商用这几份文件，请先取得原作者的授权。原作者如希望本仓库移除，请开 issue。

## 4. yaohaoliang141-max/ai-character-passport（借鉴思路）

- 上游：https://github.com/yaohaoliang141-max/ai-character-passport
- 本仓库中的位置：`src/aigc_agent/domain/storyboard/`
- 说明：借鉴了「角色护照」的数据模型和逐镜注入角色描述的做法，用 Python 重新实现，没有复制上游的代码或提示词。

---

## 许可证全文

### MIT License — 0xsline/short-drama

```
MIT License

Copyright (c) 2025 0xsline

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License — LearnPrompt/awesome-seedance

```
MIT License

Copyright (c) 2026 LearnPrompt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### CC BY 4.0

策划内容的许可全文：https://creativecommons.org/licenses/by/4.0/legalcode
