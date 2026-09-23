"""M3 上下文装配 —— 决定每次请求里到底放什么、按什么顺序放。

排布顺序（稳定 → 易变，让 prompt cache 的前缀尽可能长）：

    系统提示词 → [外部声明区] → 能力目录 → system 位 pin
      → 短期记忆(10-15 轮) → 长期记忆召回 → pre_input 位 pin → 当前输入
                              └ 每轮变，故放末尾

两处顺序是有意为之：
  · 长期记忆召回放在短期记忆之后 —— 它每轮都可能变，放前面会让后面所有
    内容的缓存失效；放末尾则只影响它自己和当前输入。
  · pre_input 位的 pin 紧贴当前输入 —— 长上下文中段召回率明显低于首尾，
    硬约束（must_not）要放在注意力最强的位置。

另：OpenAI 兼容协议下工具定义走 tools= 参数，位置由服务端决定，我们控制
不了。所以 M20 的「外部声明区」防注入措施要落在**每个工具的 description
前缀**上，而不是靠消息排序。P2 接 MCP 时在 registry 里加。

2026-09-17 加的两样（见 compaction.py）：

  · **轮内压缩**：历史轮里超过阈值的工具参数/结果折叠成存根；当前轮只保留
    最近几次迭代的原文。正文都在资产库里，留在上下文里的只是重复品。
  · **估算校准**：估算器和真实分词器对不上是常态 —— 实测 Kimi 对「JSON 里的
    中文」低估四成，18 万的熔断实际要到 25 万才触发，直接撞上模型上限。
    每次拿到真实 usage 就校准一次，之后的预算判断都按校准值算。
"""

from __future__ import annotations

from typing import Any

from ..events.bus import EventBus, EventType
from ..model.gateway import estimate_tokens
from .compaction import HARD, CompactionPolicy, fold_turn_messages
from .window import ShortTermMemory, Turn

DEFAULT_SYSTEM_PROMPT = """你是一个 AIGC 内容创作助手，服务于内部内容生产团队。

工作方式：
- 你出方案、出草稿、出候选；人做选择和终审。不要替人做最终决定。
- 关键节点默认给多个候选并说明差异，不要只给一个答案。
- 需要外部信息或要执行动作时，调用工具，不要凭空编造。
- 不确定的地方直接说不确定，不要含糊过去。

**开工前先分清是哪条产线** —— 四条的流程、配方、成本完全不同，
走错一条等于白跑，而且要到出片才看得出来：

  短剧        带剧情和对白的连续剧：剧本 → 分镜 → 资产库 → 参考图 → 逐镜生成，
              有角色、服装、跨集一致性；音频由视频模型原生生成。
  抖音短视频  30 秒左右：确认风格 → 拉真实热榜定选题 → 文案与口播分镜 → 补实拍 → 快切出片。
  广告        产品片：投放级（product-ad）或 UGC 带货（ugc-vlog），产品图当身份锁，
              上屏文字后期叠，不让生成模型画字。
  设计        海报 / 封面 / 图文卡片：生图出不带字的底图，make_poster 本地叠字。

上下文里有「当前产线」一段时，用户已经在 /type 选过，**按那条走，不要再问**。
没有的话，用户只说"做个视频""做个内容"这类没指明产线的需求，先问清楚是哪条，
不要替他选；已经说了"短剧""剧本""角色"或"热点""口播"或"广告""产品"或"海报""封面"
这类明确信号的，直接按那条走，不要多此一问。

进短剧流程后，**第一件事是调 drama_intake 判断用户给的是什么**：

  一句话想法 → 调 drama_write 写成剧本，给用户看过、他确认了，再往下拆解。
              别拿一句话直接去拆分镜，模型只能硬编，出来的和他想的不是一回事。

  完整剧本   → 问他：直接进工程，还是先按方法论扩写/修改（drama_expand）。
              后面生成很贵很慢（一集六段视频约 22 分钟），剧本不满意就往下跑最亏。

  看不准     → 如实说看不准，让他确认。不要蒙。

**短剧镜头只能通过 drama_render_shots 渲染**（它会带参考图、锁音色、做一致性校验，
渲染前先核对引用，没有引用成功的镜头不会发起生成）。某几段失败了就修好原因后再跑一次
drama_render_shots（reuse=true 只补失败的段），**不要自己改写提示词用 gen_video / gen_videos
补生成** —— 那样没有参考图，人物必然变脸；媒体层会拦下这种调用。段缺着就如实说缺，不要拼成片。

**生成模型不要自作主张换**（视频和生图都一样）：本会话锁定的模型（用户指定的）是唯一默认，
gen_video / gen_videos / gen_image / gen_images 的 model 留空就用它，不要靠 prefer 让系统自动
选型。用户要求换、或当前模型确实做不了时（比如某张图被服务商的内容护栏拒了），**可以主动提议
换一家** —— 直接带 model 调用，系统会暂停征求用户同意，采纳后才生效，并把之后的生成都切过去。
提议时要说清楚为什么换、换成哪个。不要为了省钱、更快或绕过失败悄悄换，换错模型质量和体验都很差。

**生成的东西默认落在用户当前打开的那个文件夹**（产物目录，面板和每轮的本地素材段里都写着
它在哪）：剧本、图片、视频、成片一律存那儿，不要自己另选目录、也不要散到 workspace 里。
只有用户明说了"存到 X"才写到别处；他说的是相对路径就相对产物目录解析。开工前先看一眼
那个文件夹已经有什么、缺什么，别对着满满一个目录说"没有"。

**用户的素材在他电脑上**：产物目录（生成过的剧本 .md、图片、分段视频、整集成片）和他自己放素材的
文件夹。用户提到「已有的素材 / 本地 / 文件夹 / 之前生成的」或给出路径时，**先看本地文件**：
find_materials / find_episode 找，fs_list 列目录，fs_read 读文本（含 docx），view_image / view_video
看图和视频，要进流水线先 fs_import 登记成资产。资产库（list_assets）查不到不等于没有 ——
产物目录里的文件同样算数；说「没有」之前先查一遍本地。

回答用中文，简洁直接。"""


class ContextAssembler:
    def __init__(
        self,
        bus: EventBus,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        capability_budget: int = 30_000,
        compaction: CompactionPolicy | None = None,
        token_budget: int = 0,
        calibration: float = 1.0,
    ) -> None:
        self.bus = bus
        self.system_prompt = system_prompt
        self.capability_budget = capability_budget
        self.compaction = compaction or CompactionPolicy()
        # 整个请求的 token 上限（校准后的估算）。0 = 不做预算判断。
        # 装配层给 Loop 提供 fits()，超了由 Loop 剔历史轮 —— 剔轮要走记忆提取，
        # 那是 Loop 的事，装配层不动 memory。
        self.token_budget = token_budget
        # 估算 → 真实 token 的校准系数。observe() 用真实 usage 持续修正。
        self.calibration = calibration
        self.last_estimate = 0  # 上次装配的原始估算
        self.last_tokens = 0  # 上次装配的校准估算
        self.last_folded = 0  # 上次装配折叠了几处
        # 应急档：撞上模型上限后由 Loop 置上，本轮内一直生效（新轮开始时清）
        self.shrink = False
        # 历史轮折不折的决定要记住 —— 校准系数会漂，阈值附近的轮次
        # 来回翻转会让前缀不稳定，缓存白白击穿
        self._fold_decisions: dict[int, bool] = {}

    # ---------- 校准 ----------

    def calibrated(self, raw_estimate: int) -> int:
        return int(raw_estimate * self.calibration)

    def observe(self, actual_prompt_tokens: int) -> None:
        """拿真实 usage 校准估算。指数平滑，新观测权重大 —— 上下文构成会变
        （一批工具结果进来），校准要跟得上。"""
        if actual_prompt_tokens <= 0 or self.last_estimate <= 0:
            return
        ratio = max(0.5, min(3.0, actual_prompt_tokens / self.last_estimate))
        self.calibration = 0.4 * self.calibration + 0.6 * ratio

    def fits(self, messages: list[dict[str, Any]]) -> bool:
        if not self.token_budget:
            return True
        return self.calibrated(estimate_tokens(messages)) <= self.token_budget

    def tokens_of(self, messages: list[dict[str, Any]]) -> int:
        return self.calibrated(estimate_tokens(messages))

    # ---------- 视图 ----------

    def _history_view(self, t: Turn) -> tuple[list[dict[str, Any]], int]:
        """历史轮：超过阈值的整轮折叠（参数/结果换存根），小轮原样保留。"""
        policy = HARD if self.shrink else self.compaction
        if t.index not in self._fold_decisions:
            size = self.calibrated(t.tokens or estimate_tokens(t.messages))
            self._fold_decisions[t.index] = size > policy.history_turn_tokens
        if not self._fold_decisions[t.index] and not self.shrink:
            return list(t.messages), 0
        return fold_turn_messages(t.messages, keep_tail_iterations=0, policy=policy)

    def _current_view(self, t: Turn) -> tuple[list[dict[str, Any]], int]:
        """当前轮：涨过阈值后只保留最近几次迭代的原文，更早的折叠。"""
        policy = HARD if self.shrink else self.compaction
        size = self.calibrated(estimate_tokens(t.messages))
        if size <= policy.current_turn_tokens and not self.shrink:
            return list(t.messages), 0
        return fold_turn_messages(
            t.messages, keep_tail_iterations=policy.keep_recent_iterations, policy=policy
        )

    # ---------- 装配 ----------

    async def assemble(
        self,
        memory: ShortTermMemory,
        current_turn: Turn,
        tool_catalog: str = "",
        recalled: str = "",
    ) -> list[dict[str, Any]]:
        """组装本次请求的完整消息列表。

        current_turn 已经在 memory.turns 里（它是最后一个），
        这里把它连同历史一起铺开。
        """
        messages: list[dict[str, Any]] = []
        folded = 0

        # ---- 1. 系统区（最稳定，吃缓存）----
        system_parts = [self.system_prompt]

        if tool_catalog:
            system_parts.append(
                "## 可用工具目录\n"
                "以下是你可以调用的工具。完整参数定义已随请求提供。\n\n" + tool_catalog
            )

        for p in memory.pins_at("system"):
            system_parts.append(p.content)

        messages.append({"role": "system", "content": "\n\n---\n\n".join(system_parts)})

        # ---- 2. 短期记忆（历史轮，追加为主；大轮折叠成存根）----
        live = {t.index for t in memory.turns}
        self._fold_decisions = {k: v for k, v in self._fold_decisions.items() if k in live}
        history = [t for t in memory.turns if t.index != current_turn.index]
        for t in history:
            view, n = self._history_view(t)
            folded += n
            messages.extend(view)

        # ---- 3. 长期记忆召回（每轮可能变，故置于历史之后）----
        if recalled:
            messages.append(
                {"role": "system", "content": "## 相关记忆\n" + recalled}
            )

        # ---- 4. pre_input 位 pin（注意力最强的位置）----
        pre = memory.pins_at("pre_input")
        if pre:
            messages.append(
                {
                    "role": "system",
                    "content": "## 本次必须遵守\n" + "\n\n".join(p.content for p in pre),
                }
            )

        # ---- 5. 当前轮（涨大了折旧迭代）----
        view, n = self._current_view(current_turn)
        folded += n
        messages.extend(view)

        raw = estimate_tokens(messages)
        self.last_estimate = raw
        self.last_tokens = self.calibrated(raw)
        self.last_folded = folded
        await self.bus.emit(
            EventType.CONTEXT_ASSEMBLED,
            messages=len(messages),
            turns_in_window=len(memory.turns),
            pins=len(memory.pins),
            est_tokens=raw,
            tokens=self.last_tokens,
            calibration=round(self.calibration, 2),
            folded=folded,
            shrink=self.shrink,
        )
        if folded:
            await self.bus.emit(
                EventType.CONTEXT_COMPACTED,
                folded=folded,
                tokens=self.last_tokens,
                shrink=self.shrink,
            )
        return messages
