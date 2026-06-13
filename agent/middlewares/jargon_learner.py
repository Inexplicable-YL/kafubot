"""黑话术语抽取与解释中间件。

本模块负责把已经从主对话窗口中裁剪出去的历史消息，异步转化为一组可检索的
`jargon` 记录。它的核心目标不是立即回答用户，而是持续积累“这个会话里哪些
词条可能是圈内简称、黑话、缩写或约定俗成的特殊表达”，并在证据充足时推断其
含义，供后续工具查询。

整体处理链路如下：

1. `JargonMiddleware.aafter_agent` 接收上游提供的历史消息批次。
2. 批次按 `session_id` 进入内存队列，由后台 worker 串行消费。
3. 分析模型先从消息中抽取疑似黑话词条，再和原始消息上下文重新关联。
4. 词条写入 store 后，根据出现次数阈值决定是否触发进一步语义推断。
5. 代理可通过 `query_jargon` 工具查询当前会话或关联会话中已经确认的黑话解释。

由于该模块承担的是“低优先级、后台增量学习”职责，因此它在设计上偏向稳健和可
恢复：同一会话串行处理、失败延迟重试、达到最大重试后主动丢弃最旧批次以避免
永久阻塞。
"""

import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from typing_extensions import override

from json_repair import repair_json
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from agent.base import ManagerContext, ManagerState, UserMessage
from agent.middlewares.base import (
    BaseDaemonMiddleware,
    SessionProcessOutput,
)
from agent.prompts.manager import BOT_NAME
from agent.utils import content_to_text

if TYPE_CHECKING:
    from langgraph.store.base import SearchItem

logger = logging.getLogger(__name__)

JARGON_CREATED_BY_AI = "ai"
JARGON_CREATED_BY_MANUAL = "manual"

JARGON_INFERENCE_THRESHOLDS = (4, 8, 25, 100)
"""黑话含义推断采用“逐步加证据”的策略，而不是每次命中都重新调用模型。

这几个阈值表示：当词条累计出现次数达到 4、8、25、100 时，允许触发一轮新的
语义推断。设计意图如下：

1. 4: 首次推断阈值。此时样本量刚够形成初步上下文，但仍保持较低成本。
2. 8: 第二轮校正阈值。用于在词条继续出现后，尽早修正第一次可能不稳定的结论。
3. 25: 中期稳定阈值。此时上下文通常显著增多，会同步启用采样和“参考上次含义”
   两个附加策略，控制成本并提高增量修正质量。
4. 100: 最终完成阈值。达到后会把记录标记为 `is_complete=True`，表示该词条已经
   有了足够多的观测样本，不再需要继续重复推断。
"""

JARGON_SAMPLE_RAW_CONTENT_THRESHOLDS = {25}
"""当词条在特定阈值触发推断时，如果收集到的 `raw_content` 样本过多，就对上下文做
降采样，避免把所有样本都喂给模型。当前只在 25 次阈值启用，是因为这一阶段通常
开始出现“样本已经很多，但还没必要做最终收敛”的情况；抽样一半上下文，通常就能
在成本和代表性之间取得较好的平衡。
"""

JARGON_PREVIOUS_MEANING_THRESHOLDS = {25, 100}
"""当词条在这些阈值上重新推断时，会把上一轮已经得到的 `meaning` 作为“仅供参考”
的历史结果拼进 prompt，帮助模型基于新样本做增量修正，而不是每次完全从零开始。

当前选择 25 和 100 的原因是：

1. 25: 样本量进入中期后，旧结论开始有较高复用价值。
2. 100: 最终收敛前再利用一次历史结论，便于输出更稳定的最终解释。
"""

JARGON_EXTRACTION_PROMPT = """
{messages}

请从上面这段聊天内容中提取"可能是黑话"的候选项（黑话/俚语/网络缩写/口头禅/梗/圈内简称/缩写/代称/有特殊义项的词条）。
- 必须为对话中真实出现过的短词或短语
- 必须是你无法理解含义的词语，没有明确含义的词语，请不要选择有明确含义，或者含义清晰的词语
- 排除：人名、@、表情包/图片中的内容、纯标点、常规功能词（如的、了、呢、啊等）
- 每个词条长度建议 2-8 个字符（不强制），尽量短小
- 请你提取出可能的黑话，最多30个黑话，请尽量提取所有

黑话必须为以下几种类型：
- 由字母构成的，汉语拼音首字母的简写词，例如：nb、yyds、xswl
- 英文词语的缩写，用英文字母概括一个词汇或含义，例如：CPU、GPU、API
- 中文词语的缩写，用几个汉字概括一个词汇或含义，例如：社死、内卷

提取要求：
1. 只提取词或短语，不提取完整句子。
2. `content` 必须是聊天记录里真实出现过的原文。
3. `source_id` 必须来自聊天记录中的 `<user-message source_id=...>`。
4. 普通常见词、常规成语、明显的人名地名、单个汉字、单个英文字母、单个数字、纯标点、纯表情、URL，不要提取。
5. 如果一个词只是字面义，没有明显特殊含义，也不要提取。
6. 允许同一个 `content` 对应多个 `source_id`，请分别输出多项。

输出要求：
将黑话以 JSON 数组输出，每个元素为一个对象，结构如下（注意字段名）：
注意请若一个词条同时出现在多个聊天记录中（有多个 source_id），请分别输出多个词条，content 允许并推荐重复，但是不要输出两条一模一样的 JSON 对象。

[
  {{"content": "词条", "source_id": "12"}},
  {{"content": "词条2", "source_id": "5"}},
  {{"content": "词条2", "source_id": "8"}}
]

黑话jargon条目：
- content:表示黑话的内容
- source_id：该黑话对应的来源编号，即上方聊天记录中的 source_id 数字（例如 `<user-message source_id=3>`。 对应 "3"），请只输出数字本身

输出 JSON：
""".strip()

JARGON_INFERENCE_WITH_CONTEXT_PROMPT = """
**词条内容**
{content}
**词条出现的上下文。其中的{bot_name}的发言内容是你自己的发言**
{raw_content_text}
{previous_meaning_section}

请根据上下文，推断"{content}"这个词条的含义。
- 如果这是一个黑话、俚语或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明
- {bot_name} 的发言内容可能包含错误，请不要参考其发言内容
- 如果上下文信息不足，无法推断含义，请设置 no_info 为 true
{previous_meaning_instruction}

以 JSON 格式输出：
{{
  "meaning": "详细含义说明（包含使用场景、来源、具体解释等）",
  "no_info": false
}}
注意：如果信息不足无法推断，请设置 "no_info": true，此时 meaning 可以为空字符串
""".strip()

JARGON_INFERENCE_CONTENT_ONLY_PROMPT = """
**词条内容**
{content}

请仅根据这个词条本身，推断其含义。
- 如果这是一个黑话、俚语或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明

以 JSON 格式输出：
{{
  "meaning": "详细含义说明（包含使用场景、来源、具体解释等）"
}}
""".strip()

JARGON_COMPARE_INFERENCE_PROMPT = """
**推断结果1（基于上下文）**
{inference1}

**推断结果2（仅基于词条）**
{inference2}

请比较这两个推断结果，判断它们是否相同或类似。
请忽略细微的差别，关注主要含义是否相似

以 JSON 格式输出：
{{
  "is_similar": true/false,
  "reason": "判断理由"
}}
""".strip()


class QueryJargonInput(BaseModel):
    """`query_jargon` 工具的输入模型。

    Attributes:
        words: 需要查询含义的词条列表。调用方通常会传入用户当前提到、但模型
            自身无法可靠解释的简称、黑话或缩写。
    """

    words: list[str] = Field(description="要查询的词条列表。")


class JargonEntry(TypedDict):
    """抽取得到的黑话候选条目。

    Attributes:
        content: 候选词条本身。它应当是模型从聊天记录中抽取出的短词或短语。
        raw_content: 与该词条关联的一组原始上下文文本。这里会按集合去重，避免
            同一来源片段在后续入库和推断时被重复计算。
    """

    content: str
    raw_content: set[str]


class JargonRecord(TypedDict):
    """持久化到 store 中的标准黑话记录。

    Attributes:
        content: 黑话或候选词条的标准文本。
        raw_content: 收集到的上下文样本列表。每一项通常是一段围绕该词条的消息
            窗口，用于后续含义推断。
        session_id_dict: 记录该词条在不同会话中的出现次数。键为 `session_id`，
            值为该会话下累计命中的次数。
        session_id: 该记录首次归属或当前主要归属的会话 ID。对于会话级词条，
            这个字段通常就是创建它的会话。
        is_global: 是否允许该记录被视为全局共享词条。为 `True` 时，即使当前
            会话不在 `session_id_dict` 内，也可以在查询时作为较低优先级候选。
        count: 该词条被累计观察到的总次数。它直接驱动是否进入下一轮语义推断。
        meaning: 推断出的黑话含义。只有在 `is_jargon` 为 `True` 时通常才会保留。
        is_jargon: 当前记录是否已被判定为“确实是黑话或特殊术语”。
        is_complete: 该记录是否已经达到最终推断阈值，不再需要继续重复推断。
        last_inference_count: 上一次执行含义推断时，该记录对应的 `count` 值。
            用于避免同一阈值区间内的重复推断。
        created_by: 记录创建来源。`manual` 表示人工维护，`ai` 表示自动抽取。
            人工记录在更新和查询排序上具有更高优先级。
        created_at: 记录创建时间，使用 ISO 8601 字符串。
        updated_at: 记录最近更新时间，使用 ISO 8601 字符串。
    """

    content: str
    raw_content: list[str]
    session_id_dict: dict[str, int]
    session_id: str
    is_global: bool
    count: int
    meaning: str
    is_jargon: bool
    is_complete: bool
    last_inference_count: int
    created_by: Literal["ai", "manual"]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class PendingJargonAnalysisBatch:
    """等待后台分析的单个消息批次。

    Attributes:
        session_id: 该批次所属的会话 ID。
        messages: 当前批次包含的消息列表，通常来自上游已经裁剪出的历史消息。
    """

    session_id: str
    messages: list[BaseMessage]


_DEFAULT_NAMESPACE = "jargon"


class JargonLearnerMiddleware(BaseDaemonMiddleware[PendingJargonAnalysisBatch]):
    """为代理提供异步黑话抽取、入库、推断与查询能力。

    这个中间件刻意不把黑话分析放在主对话的同步路径里，而是在消息被上游裁剪后
    再异步处理。这样可以把较重的术语学习成本挪到后台，同时尽量不影响当前轮响
    应时延。

    模块内部把整条链路拆成三个层次：

    1. 收集层：在 `aafter_agent` 中接收裁剪后的历史消息，并按会话写入待处理队列。
    2. 工作层：后台 worker 串行分析每个会话，抽取候选词条并更新持久化记录。
    3. 查询层：通过 `query_jargon` 工具向主代理暴露已确认黑话的含义查询能力。

    作用域命中时遵循稳定的优先级策略：人工记录优先于 AI 记录，当前会话优先于
    关联会话，关联会话优先于全局共享记录；在同优先级下，再按出现次数选择更稳
    定的结果。

    Attributes:
        tools: 暴露给代理的工具列表。目前包含 `query_jargon`。
        namespace_root: 存储黑话记录时使用的根命名空间。
    """

    state_schema = ManagerState

    def __init__(
        self,
        analyze_model: BaseChatModel,
        *,
        max_sessions: int = 100,
        max_input_chars: int = 18000,
        max_raw_content_chars: int = 600,
        max_batches: int = 3,
        context_window_size: int = 3,
        max_retries: int = 3,
        store: BaseStore | None = None,
        jargon_group_resolver: Callable[[str], set[str] | tuple[set[str], bool]]
        | None = None,
    ) -> None:
        """初始化黑话中间件。

        Args:
            analyze_model: 用于黑话抽取与含义推断的分析模型。
            max_sessions: 内存中最多同时维护多少个活跃会话的队列、重试状态和缓存。
                超出后由 `LRUCache` 自动驱逐较久未使用的会话状态。
            max_input_chars: 单次发给黑话抽取模型的最大输入字符数。
            max_raw_content_chars: 单条上下文样本允许保留的最大字符数。它既用于
                存储去重，也用于控制含义推断时的上下文规模。
            max_batches: 单次后台分析最多合并多少个待处理批次。该值越大，单次
                模型调用上下文越完整，但时延和 token 成本也会提高。
            context_window_size: 生成上下文窗口时，目标用户消息前后各保留多少条
                邻近消息，用于帮助推断词条含义。
            max_retries: 同一会话的分析任务最多允许失败重试多少次。
            store: 可选的持久化存储。若未显式传入，会在运行时优先使用
                `runtime.store`。建议使用独立的 store 实例，以避免与主代理的
                其他数据产生混淆。
            jargon_group_resolver: 会话作用域解析器。它用于把一个 `session_id`
                扩展为一组“可共享黑话词库”的关联会话 ID，也可额外返回是否存在
                全局共享语义。
        """
        super().__init__(
            max_sessions=max_sessions,
            max_retries=max_retries,
            max_batch_window_size=max_batches,
            logger_config={
                "worker_name": "JargonLearner",
                "job_name": "jargon learning",
                "process_failure_log": "Failed to analyze jargon entries",
                "retry_exhausted_label": "Jargon learning",
            },
        )
        self.analyze_model = analyze_model
        self.max_input_chars = max_input_chars
        self.max_raw_content_chars = max_raw_content_chars
        self.context_window_size = context_window_size
        self._store = store
        self.jargon_group_resolver = jargon_group_resolver

        self._max_message_chars = max(1, max_input_chars // max_batches)
        self.namespace_root = _DEFAULT_NAMESPACE

        self.tools = [
            tool(
                "query_jargon",
                description="查询当前聊天上下文中的黑话或词条含义。用法：当你认为某些词的含义不明确，或者用户询问某些词的含义，需要进行查询。",
                args_schema=QueryJargonInput,
            )(self._query_jargon_tool)
        ]

    @override
    async def aafter_agent(
        self,
        state: ManagerState,
        runtime: Runtime[ManagerContext],
    ) -> dict[str, Any] | None:
        """接收裁剪后的历史消息，并把它们异步送入黑话分析队列。

        该钩子只处理 `summary_pruned_messages`，也就是已经从主上下文窗口移除的
        历史消息。这样做有两个好处：

        1. 不会影响当前轮主推理的 token 预算。
        2. 黑话学习可以和摘要系统一样，作为一种后台增量知识沉淀机制。

        Args:
            state: 当前代理状态。这里会读取其中的 `summary_pruned_messages`。
            runtime: LangGraph 运行时上下文，用于获取 `session_id` 和 store。

        Returns:
            始终返回 `None`，因为该钩子只负责排队副作用，不直接改写状态。
        """
        if runtime.store is not None:
            self._store = self._store or runtime.store
        if messages := cast(
            "list[BaseMessage] | None", state.get("summary_pruned_messages")
        ):
            session_id = runtime.context["session_id"]
            await self._enqueue_batch(
                session_id,
                PendingJargonAnalysisBatch(
                    session_id=session_id,
                    messages=list(messages),
                ),
            )
        return None

    @override
    async def process_batches(
        self,
        session_id: str,
        batches: tuple[PendingJargonAnalysisBatch, ...],
    ) -> SessionProcessOutput:
        """分析单个会话当前积压的黑话批次。

        父类已经负责从会话队列中切出当前批次窗口，这里只处理本轮业务逻辑：
        构造抽取输入，然后执行“抽取候选词条 -> 入库 -> 必要时触发含义推断”。

        Args:
            session_id: 当前会话 ID。
            batches: 父类传入的当前批次窗口。

        Returns:
            业务处理结果。父类会统一归一化返回值、出队和处理失败重试。
        """
        if self._store is None:
            logger.warning(
                "Skipping jargon analysis for session %s because store is unavailable",
                session_id,
            )
            return None

        batch_count, messages_text, source_map = self._build_extraction_input(batches)
        if batch_count <= 0:
            return None

        if not messages_text:
            return batch_count

        entries = await self._extract_entries(messages_text, source_map)
        if entries is None:
            return 0, True

        await self._process_entries(
            session_id=session_id,
            entries=entries,
            store=self._store,
        )
        return batch_count

    def _build_extraction_input(
        self,
        batches: Sequence[PendingJargonAnalysisBatch],
    ) -> tuple[int, str, dict[str, str]]:
        """把若干待处理批次压缩为一次抽取模型输入。

        该方法一边构建发给模型的消息文本，一边维护后续回查上下文所需的
        `source_id -> context` 映射。消息拼装时只会保留用户消息和机器人消息，
        并使用带标签的文本格式提示模型进行抽取。

        Args:
            batches: 父类已经切好的当前批次窗口。

        Returns:
            一个三元组 `(batch_count, messages_text, source_map)`，其中
            `batch_count` 是本次纳入分析的批次数，`messages_text` 是发给抽取模型
            的文本，`source_map` 用于把模型返回的 `source_id` 重新映射回上下文
            窗口。
        """
        selected_lines: list[str] = []
        selected_records: list[dict[str, Any]] = []
        batch_count = 0
        synthetic_index = 0

        for batch in batches:
            batch_records: list[dict[str, Any]] = []

            for message in batch.messages:
                if isinstance(message, HumanMessage) and isinstance(
                    raw := message.additional_kwargs.get("raw"), UserMessage
                ):
                    if any(m.type in {"image", "meme"} for m in raw.message):
                        continue
                    plain = _clean_text(raw.as_plain_content())
                    if not plain:
                        continue

                    source_id = (
                        _clean_text(raw.message_id)
                        or f"synthetic-{batch_count}-{synthetic_index}"
                    )
                    synthetic_index += 1
                    if raw.message_id:
                        extraction_line = raw.as_content()
                    else:
                        extraction_line = (
                            f'<user-message source_id="{source_id}">\n'
                            f"{plain}\n"
                            "</user-message>"
                        )

                    batch_records.append(
                        {
                            "line": _truncate(extraction_line, self._max_message_chars),
                            "plain": plain,
                            "source_id": source_id,
                            "is_user_source": True,
                        }
                    )
                    continue

            if not batch_records:
                batch_count += 1
                continue

            batch_lines = [str(record["line"]) for record in batch_records]
            candidate = "\n".join([*selected_lines, *batch_lines]).strip()
            if selected_lines and len(candidate) > self.max_input_chars:
                break

            selected_lines.extend(batch_lines)
            selected_records.extend(batch_records)
            batch_count += 1

            if len(candidate) > self.max_input_chars:
                messages_text = _truncate(candidate, self.max_input_chars)
                return (
                    batch_count,
                    messages_text,
                    self._build_source_context_map(selected_records),
                )

        return (
            batch_count,
            "\n".join(selected_lines).strip(),
            self._build_source_context_map(selected_records),
        )

    def _build_source_context_map(
        self, records: list[dict[str, Any]]
    ) -> dict[str, str]:
        """为每条用户消息构建可回查的上下文窗口。

        Args:
            records: `_build_extraction_input` 收集到的线性消息记录。

        Returns:
            从 `source_id` 到其邻近上下文文本的映射。只有用户消息会被纳入该映射，
            因为黑话候选的来源最终必须对应到用户消息。
        """
        source_map: dict[str, str] = {}
        for index, record in enumerate(records):
            source_id = _clean_text(record.get("source_id"))
            if not source_id or not record.get("is_user_source"):
                continue

            start = max(0, index - self.context_window_size)
            end = min(len(records), index + self.context_window_size + 1)
            context_lines: list[str] = []
            for display_index, context_record in enumerate(records[start:end], start=1):
                plain = _clean_text(context_record.get("plain"))
                if plain:
                    context_lines.append(f"[{display_index}] {plain}")

            context_text = "\n".join(context_lines).strip()
            if context_text:
                source_map[source_id] = _truncate(
                    context_text,
                    self.max_raw_content_chars,
                )
        return source_map

    async def _extract_entries(
        self,
        messages_text: str,
        source_map: dict[str, str],
    ) -> list[JargonEntry] | None:
        """调用分析模型抽取黑话候选，并把结果合并到上下文样本集合。

        Args:
            messages_text: 已格式化好的聊天记录文本。
            source_map: 从 `source_id` 到上下文窗口文本的映射。

        Returns:
            抽取得到的黑话候选列表。若模型调用失败或返回结果无法解析，则返回
            `None`，由上层决定是否重试。
        """
        try:
            response = await self.analyze_model.ainvoke(
                JARGON_EXTRACTION_PROMPT.format(messages=messages_text),
                config={"metadata": {"lc_source": "jargon_extract"}},
            )
        except Exception:
            logger.exception("Failed to extract jargon entries")
            return None

        parsed = _parse_result(content_to_text(response.content))
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list) and any(
                    isinstance(item, dict) for item in value
                ):
                    parsed = value
                    break
        if not isinstance(parsed, list):
            logger.warning("Failed to parse jargon extraction result")
            return None

        merged_entries: dict[str, set[str]] = defaultdict(set)
        for item in parsed:
            if not isinstance(item, dict):
                continue
            content = _clean_text(item.get("content"))
            source_id = _clean_text(item.get("source_id"))
            if content and source_id and source_id in source_map:
                merged_entries[content].add(source_map[source_id])

        return [
            {"content": content, "raw_content": raw_content}
            for content, raw_content in merged_entries.items()
            if raw_content
        ]

    async def _process_entries(  # noqa: PLR0915
        self,
        *,
        session_id: str,
        entries: list[JargonEntry],
        store: BaseStore,
    ) -> tuple[int, int]:
        """将抽取出的候选词条合并、入库并触发必要的含义推断。

        这里承担了模块里最核心的“增量归档”职责：

        1. 过滤明显无效的候选词条。
        2. 合并同名词条的多份上下文样本。
        3. 在当前会话、关联会话和全局记录中查找最佳匹配项。
        4. 更新命中次数和上下文，或创建新记录。
        5. 对达到阈值的记录安排下一阶段含义推断。

        Args:
            session_id: 当前正在处理的会话 ID。
            entries: 模型抽取出的黑话候选集合。
            store: 用于持久化黑话记录的存储后端。

        Returns:
            一个二元组 `(saved, updated)`，分别表示新建记录数和更新记录数。
        """
        merged_entries: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            content = _clean_text(entry["content"])
            if _is_invalid_jargon_candidate(content):
                continue
            for raw_content in entry["raw_content"]:
                raw_content = _truncate(
                    _clean_text(raw_content), self.max_raw_content_chars
                )
                if raw_content:
                    merged_entries[content].add(raw_content)

        if not merged_entries:
            return 0, 0

        related_session_ids, has_global_share = self._resolve_jargon_scope(session_id)
        namespace_prefix = (self.namespace_root,)
        saved = 0
        updated = 0
        pending_inference: list[tuple[tuple[str, ...], str, JargonRecord]] = []

        for content, raw_content in merged_entries.items():
            namespace = (self.namespace_root, "sessions", session_id)
            key = content
            matched_item: SearchItem | None = None
            matched_record: JargonRecord | None = None
            matched_rank: tuple[int, int, int] | None = None

            matched_candidates: list[SearchItem] = []
            offset = 0

            while True:
                items = await store.asearch(
                    namespace_prefix,
                    query=None,
                    filter={"content": content},
                    limit=100,
                    offset=offset,
                )
                if not items:
                    break

                matched_candidates.extend(items)

                if len(items) < 100:
                    break

                offset += 100

            for item in matched_candidates:
                record = _normalize_record(item.value)
                if record is None or record["content"] != content:
                    continue

                scope_rank = _record_scope_rank(
                    record, session_id, related_session_ids, has_global_share
                )
                if scope_rank is None:
                    continue

                rank = (
                    0 if record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                    scope_rank,
                    -record["count"],
                )
                if matched_rank is None or rank < matched_rank:
                    matched_item = item
                    matched_record = record
                    matched_rank = rank

            now = datetime.now(UTC).isoformat()

            if matched_record is None:
                record = cast(
                    "JargonRecord",
                    {
                        "content": content,
                        "raw_content": sorted(raw_content),
                        "session_id_dict": {session_id: 1},
                        "session_id": session_id,
                        "is_global": False,
                        "count": 1,
                        "meaning": "",
                        "is_jargon": False,
                        "is_complete": False,
                        "last_inference_count": 0,
                        "created_by": JARGON_CREATED_BY_AI,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                saved += 1
            elif matched_item is not None:
                record = matched_record
                namespace = matched_item.namespace
                key = matched_item.key
                if record["created_by"] == JARGON_CREATED_BY_MANUAL:
                    logger.debug("Jargon '%s' is manual; skip AI update", content)
                    continue
                record["count"] += 1
                record["raw_content"] = sorted(
                    set(record["raw_content"]).union(raw_content)
                )
                record["session_id_dict"][session_id] = (
                    record["session_id_dict"].get(session_id, 0) + 1
                )
                record["updated_at"] = now
                updated += 1
            else:
                continue

            await store.aput(namespace, key, dict(record), index=False)
            if self._should_infer_meaning(record):
                pending_inference.append((namespace, key, record))

        logger.info("[%s]疑似黑话: %s", session_id, ",".join(merged_entries))
        for namespace, key, record in pending_inference:
            await self._infer_meaning(namespace, key, record, store)

        return saved, updated

    async def _infer_meaning(
        self,
        namespace: tuple[str, ...],
        key: str,
        record: JargonRecord,
        store: BaseStore,
    ) -> None:
        """对已达到阈值的候选词条执行多阶段含义推断。

        推断流程分为三步：

        1. 基于上下文推断词义。
        2. 仅基于词条字面形式推断词义。
        3. 比较两次推断是否相近，从而判断它更像普通词汇还是黑话。

        该设计的核心目的，是降低“把常见词误判为黑话”的概率。若一个词即使脱离
        上下文也能得到和上下文推断近似的解释，通常说明它并不是强依赖语境的圈内
        术语。

        Args:
            namespace: 记录当前所在的存储命名空间。
            key: 记录在 store 中的键。
            record: 需要推断含义的黑话记录。
            store: 持久化黑话记录的存储后端。
        """
        if record["created_by"] == JARGON_CREATED_BY_MANUAL:
            return

        raw_content_list = [item for item in record["raw_content"] if _clean_text(item)]
        if not raw_content_list:
            logger.warning(
                "Jargon '%s' has no raw_content; skip inference", record["content"]
            )
            return

        if (
            record["count"] in JARGON_SAMPLE_RAW_CONTENT_THRESHOLDS
            and len(raw_content_list) > 1
        ):
            raw_content_list = random.sample(
                raw_content_list,
                max(1, len(raw_content_list) // 2),
            )
        raw_content_text = "\n".join(raw_content_list)

        previous_meaning_section = ""
        previous_meaning_instruction = ""
        if (
            record["count"] in JARGON_PREVIOUS_MEANING_THRESHOLDS
            and record["meaning"].strip()
        ):
            previous_meaning_section = (
                f"\n**上一次推断的含义（仅供参考）**\n{record['meaning']}"
            )
            previous_meaning_instruction = "- 请参考上一次推断的含义，结合新的上下文信息，给出更准确或更新的推断结果"

        async def ask(prompt: str, source: str) -> dict[str, Any] | None:
            try:
                response = await self.analyze_model.ainvoke(
                    prompt,
                    config={"metadata": {"lc_source": f"jargon_{source}"}},
                )
            except Exception:
                logger.exception("Failed jargon inference call: %s", source)
                return None
            result = _parse_result(content_to_text(response.content))
            if not isinstance(result, dict):
                logger.warning("Jargon inference result is not a JSON object")
                return None
            return result

        inference1 = await ask(
            JARGON_INFERENCE_WITH_CONTEXT_PROMPT.format(
                bot_name=BOT_NAME,
                content=record["content"],
                raw_content_text=raw_content_text,
                previous_meaning_section=previous_meaning_section,
                previous_meaning_instruction=previous_meaning_instruction,
            ),
            "inference_with_context",
        )
        if not inference1:
            return

        if inference1.get("no_info") or not _clean_text(inference1.get("meaning")):
            record["last_inference_count"] = record["count"]
            record["updated_at"] = datetime.now(UTC).isoformat()
            await store.aput(namespace, key, dict(record), index=False)
            return

        inference2 = await ask(
            JARGON_INFERENCE_CONTENT_ONLY_PROMPT.format(content=record["content"]),
            "inference_content_only",
        )
        if not inference2:
            return

        comparison = await ask(
            JARGON_COMPARE_INFERENCE_PROMPT.format(
                inference1=json.dumps(inference1, ensure_ascii=False),
                inference2=json.dumps(inference2, ensure_ascii=False),
            ),
            "inference_compare",
        )
        if not comparison:
            return

        record["is_jargon"] = not bool(comparison.get("is_similar"))
        record["meaning"] = (
            _clean_text(inference1.get("meaning")) if record["is_jargon"] else ""
        )
        record["last_inference_count"] = record["count"]
        record["is_complete"] = record["count"] >= JARGON_INFERENCE_THRESHOLDS[-1]
        record["updated_at"] = datetime.now(UTC).isoformat()
        await store.aput(namespace, key, dict(record), index=False)

        if record["is_jargon"]:
            logger.info(
                "[黑话]%s的含义是 %s",
                record["content"],
                record["meaning"] or "无详细说明",
            )
        else:
            logger.info("[%s]%s 不是黑话", record["session_id"], record["content"])

    async def _query_jargon(
        self,
        runtime: ToolRuntime[ManagerContext, ManagerState],
        keyword: str,
        limit: int = 10,
        case_sensitive: bool = False,
        fuzzy: bool = True,
    ) -> list[dict[str, str]]:
        """在当前可见作用域内查询黑话词条。

        Args:
            runtime: 工具运行时上下文，用于获取当前会话和 store。
            keyword: 要查找的关键词。
            limit: 最多返回多少条结果。
            case_sensitive: 是否区分大小写。
            fuzzy: 是否启用包含式模糊匹配。为 `False` 时执行精确匹配。

        Returns:
            匹配结果列表。每个元素都包含 `content` 和 `meaning` 两个字段，且只
            返回已经被判定为黑话、并拥有非空含义说明的记录。
        """
        keyword = _clean_text(keyword)
        store = self._store or runtime.store
        if not keyword or store is None:
            return []

        namespace = (self.namespace_root,)
        session_id = runtime.context["session_id"]
        related_session_ids, has_global_share = self._resolve_jargon_scope(session_id)
        keyword_cmp = keyword if case_sensitive else keyword.casefold()
        results: list[tuple[int, SearchItem, JargonRecord]] = []
        offset = 0
        while True:
            if fuzzy:
                items = await store.asearch(
                    namespace,
                    query=None,
                    filter={
                        "is_jargon": True,
                        "meaning": {"$ne": ""},
                    },
                    limit=100,
                    offset=offset,
                )
            else:
                items = await store.asearch(
                    namespace,
                    query=None,
                    filter={
                        "content": keyword,
                        "is_jargon": True,
                        "meaning": {"$ne": ""},
                    },
                    limit=100,
                    offset=offset,
                )
            if not items:
                break

            for item in items:
                record = _normalize_record(item.value)
                if (
                    record is None
                    or not record["is_jargon"]
                    or not record["meaning"].strip()
                ):
                    continue

                scope_rank = _record_scope_rank(
                    record, session_id, related_session_ids, has_global_share
                )
                if scope_rank is None:
                    continue

                content_cmp = (
                    record["content"]
                    if case_sensitive
                    else record["content"].casefold()
                )
                if (fuzzy and keyword_cmp in content_cmp) or (
                    not fuzzy and keyword_cmp == content_cmp
                ):
                    results.append((scope_rank, item, record))

            if len(items) < 100:
                break
            offset += 100

        deduped: dict[str, tuple[int, SearchItem, JargonRecord]] = {}
        for scope, item, record in results:
            current = deduped.get(record["content"])
            rank = (
                0 if record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                scope,
                -record["count"],
            )
            if current is None:
                deduped[record["content"]] = (scope, item, record)
                continue
            current_scope, _, current_record = current
            current_rank = (
                0 if current_record["created_by"] == JARGON_CREATED_BY_MANUAL else 1,
                current_scope,
                -current_record["count"],
            )
            if rank < current_rank:
                deduped[record["content"]] = (scope, item, record)

        sorted_results = sorted(
            deduped.values(),
            key=lambda item: (
                item[0],
                item[2]["created_by"] != JARGON_CREATED_BY_MANUAL,
                -item[2]["count"],
            ),
        )
        return [
            {"content": record["content"], "meaning": record["meaning"]}
            for _, _, record in sorted_results[:limit]
        ]

    async def _query_jargon_tool(
        self,
        runtime: ToolRuntime[ManagerContext, ManagerState],
        words: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """`query_jargon` 工具的实现入口。

        它会先规范化并去重输入词条，再按“精确匹配优先、失败后退化到模糊匹配”的
        顺序逐个查询。

        Args:
            runtime: 工具运行时上下文。
            words: 调用方请求解释的一组词条。

        Returns:
            形如 `{"results": [...]}` 的结构化结果，其中每个元素包含原始查询词、
            是否命中以及命中的候选解释列表。
        """
        normalized_words = []
        seen: set[str] = set()
        for word in words if isinstance(words, list) else []:
            word_text = _clean_text(word)
            if not word_text or word_text in seen:
                continue
            normalized_words.append(word_text)
            seen.add(word_text)

        results: list[dict[str, Any]] = []
        for word in normalized_words:
            exact_matches = await self._query_jargon(
                runtime,
                keyword=word,
                limit=5,
                case_sensitive=False,
                fuzzy=False,
            )
            matched_entries = exact_matches
            if not matched_entries:
                matched_entries = await self._query_jargon(
                    runtime,
                    keyword=word,
                    limit=5,
                    case_sensitive=False,
                    fuzzy=True,
                )

            results.append(
                {
                    "word": word,
                    "found": bool(matched_entries),
                    "matches": matched_entries,
                }
            )

        return {"results": results}

    def _resolve_jargon_scope(self, session_id: str) -> tuple[set[str], bool]:
        """解析当前会话可见的黑话作用域。

        Args:
            session_id: 当前会话 ID。

        Returns:
            一个二元组 `(related_session_ids, has_global_share)`。其中第一项表示与
            当前会话共享黑话词库的所有会话 ID 集合；第二项表示调用方是否声明了
            额外的全局共享语义。
        """
        related_session_ids = {session_id} if session_id else set()
        has_global_share = False
        if self.jargon_group_resolver is None or not session_id:
            return related_session_ids, has_global_share

        try:
            resolved = self.jargon_group_resolver(session_id)
        except Exception:
            logger.exception("Failed to resolve jargon group scope for %s", session_id)
            return related_session_ids, has_global_share

        if isinstance(resolved, tuple):
            raw_ids, has_global_share = resolved
        else:
            raw_ids = resolved

        related_session_ids.update(str(item) for item in raw_ids if str(item).strip())
        return related_session_ids, bool(has_global_share)

    @staticmethod
    def _should_infer_meaning(record: JargonRecord) -> bool:
        """判断某条记录当前是否应当触发新一轮含义推断。

        Args:
            record: 待判断的黑话记录。

        Returns:
            若记录达到下一个推断阈值、尚未完成且不是人工记录，则返回 `True`。
        """
        if (
            record["created_by"] == JARGON_CREATED_BY_MANUAL
            or record["is_complete"]
            or record["count"] < JARGON_INFERENCE_THRESHOLDS[0]
            or record["count"] <= record["last_inference_count"]
        ):
            return False

        next_threshold = next(
            (
                threshold
                for threshold in JARGON_INFERENCE_THRESHOLDS
                if threshold > record["last_inference_count"]
            ),
            None,
        )
        return next_threshold is not None and record["count"] >= next_threshold

    @override
    async def on_close(self) -> None:
        self._store = None


def _normalize_record(value: Any) -> JargonRecord | None:
    """把 store 中的原始值规范化为 `JargonRecord`。

    该函数兼容若干宽松输入形式，例如 `raw_content` 或 `session_id_dict` 被序列化
    成 JSON 字符串的情况。

    Args:
        value: store 返回的原始对象。

    Returns:
        规范化后的 `JargonRecord`。若输入不是可识别的记录结构，则返回 `None`。
    """
    if not isinstance(value, dict):
        return None

    content = _clean_text(value.get("content"))
    if not content:
        return None

    raw_content = value.get("raw_content") or []
    if isinstance(raw_content, str):
        try:
            raw_content = json.loads(raw_content)
        except json.JSONDecodeError:
            raw_content = [raw_content]

    session_id_dict = value.get("session_id_dict") or {}
    if isinstance(session_id_dict, str):
        try:
            session_id_dict = json.loads(session_id_dict)
        except json.JSONDecodeError:
            session_id_dict = {}

    return {
        "content": content,
        "raw_content": sorted(
            {_clean_text(item) for item in raw_content if _clean_text(item)}
        ),
        "session_id_dict": {
            str(key): int(raw or 0) for key, raw in dict(session_id_dict).items()
        },
        "session_id": str(value.get("session_id") or ""),
        "is_global": bool(value.get("is_global", False)),
        "count": int(value.get("count") or 0),
        "meaning": _clean_text(value.get("meaning")),
        "is_jargon": bool(value.get("is_jargon", False)),
        "is_complete": bool(value.get("is_complete", False)),
        "last_inference_count": int(value.get("last_inference_count") or 0),
        "created_by": cast(
            "Literal['ai', 'manual']",
            str(value.get("created_by") or JARGON_CREATED_BY_AI).casefold(),
        ),
        "created_at": str(value.get("created_at") or ""),
        "updated_at": str(value.get("updated_at") or ""),
    }


def _record_scope_rank(
    record: JargonRecord,
    session_id: str,
    related_session_ids: set[str],
    has_global_share: bool,
) -> int | None:
    """计算某条记录相对当前会话的作用域优先级。

    Args:
        record: 待评估的黑话记录。
        session_id: 当前会话 ID。
        related_session_ids: 与当前会话共享黑话作用域的会话 ID 集合。

    Returns:
        当前会话命中时返回 `0`，关联会话命中时返回 `1`，全局记录返回 `2`；
        若该记录对当前会话不可见，则返回 `None`。
    """
    record_session_ids = set(record["session_id_dict"])
    if session_id and session_id in record_session_ids:
        return 0
    if related_session_ids.intersection(record_session_ids):
        return 1
    if record["is_global"] or has_global_share:
        return 2
    return None


def _parse_result(text: str) -> Any | None:
    """从模型响应文本中尽量稳健地提取 JSON 结果。

    该函数会依次处理以下常见情况：

    1. 结果被包裹在 Markdown 代码块中。
    2. 文本前后混入了说明性自然语言。
    3. JSON 存在轻微格式错误，需要借助 `json_repair` 自动修复。

    Args:
        text: 模型返回的原始文本。

    Returns:
        解析成功时返回对应的 Python 对象，否则返回 `None`。
    """
    raw = text.strip()
    if match := re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE):
        raw = match[1].strip()
    else:
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    candidates = [raw]
    for left, right in (("{", "}"), ("[", "]")):
        start = raw.find(left)
        end = raw.rfind(right)
        if start != -1 and end != -1 and start < end:
            candidate = raw[start : end + 1].strip()
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.debug("Failed to parse jargon JSON candidate directly")

        try:
            repaired = repair_json(candidate)
            if isinstance(repaired, tuple):
                repaired = repaired[0]
            if not isinstance(repaired, str):
                repaired = json.dumps(repaired, ensure_ascii=False)
            return json.loads(repaired)
        except Exception:
            logger.debug("Failed to repair jargon JSON candidate", exc_info=True)
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_invalid_jargon_candidate(content: str) -> bool:
    content = _clean_text(content)
    if not content or (
        len(content) == 1
        and (
            "\u4e00" <= content <= "\u9fff"
            or "a" <= content <= "z"
            or "A" <= content <= "Z"
            or "0" <= content <= "9"
        )
    ):
        return True
    if "SELF" in content:
        return True
    bot_name = _clean_text(BOT_NAME)
    if bot_name and bot_name in content:
        return True
    if re.fullmatch(r"https?://\S+", content, re.IGNORECASE):
        return True
    if re.fullmatch(r"[\W_]+", content):
        return True
    return bool(":meme" in content or ":image" in content)


__all__ = ["JargonLearnerMiddleware"]
