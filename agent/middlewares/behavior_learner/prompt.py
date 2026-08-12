BEHAVIOR_LEARN_PROMPT = """
{chat_str}

你是 {bot_name} 的行为表现学习器。学习前，系统已经先把同一批聊天拆成 1 到 5 个行为场景片段：
{scene_profile}

请从后续真实聊天消息中，抽取各个场景片段下可复用的“场景-行为-结果”模式。这里学习的不是说话风格，而是人在某种场景中采取的行为策略，以及它带来的结果。

本系统有两条行为学习通道：
- 观察学习：学习其他用户或群体在某个场景下做了什么，以及产生了什么结果。用于理解人类互动模式。
- 自身反馈：学习 {bot_name} 自己在某个场景下做了什么，以及产生了什么结果。用于后续反馈、修正和强化 {bot_name} 的已有动作。

要求：
- 按格式输出
- segment_id 必须来自上面的场景片段；如果无法判断，选择 source_ids 重叠最多的片段。
- actor_type 只能是 "other_user"、"group_collective"、"maibot_self" 或 "unknown"。
- learning_type 只能是 "observed_behavior" 或 "self_reflection"。
- speaker=SELF 代表 {bot_name} 自己。如果行为主体是 {bot_name}，actor_type 必须写 "maibot_self"，learning_type 必须写 "self_reflection"。
- 如果行为主体是其他用户，actor_type 写 "other_user"，learning_type 写 "observed_behavior"。
- 如果行为是多名用户共同形成的群体互动，actor_type 写 "group_collective"，learning_type 写 "observed_behavior"。
- 只抽取已经能从连续聊天中看出因果链的模式。
- action 写可执行、可复用的行为表现；outcome 写对话中已经出现或高度可推断的后果。
- action 应抽象到“相似场景下还能再次执行”的粒度，保留行为结构，不要绑定一次性对象、具体型号、临时梗词、截图内容或某个专名。
- outcome 应描述可观察的互动变化，例如“补充关键信息”“情绪缓和”“继续追问”“话题转向轻松玩梗”，不要只写“氛围活跃”这种泛泛结果。
- 如果两条候选只有具体对象不同、行为结构相同，优先合成一条更抽象的行为模式。
- 如果一段消息只是具体事实、资料、配置值、查询结果或一次性任务，不要作为行为模式学习；只有其中体现了可复用互动策略时才抽取。
- 可以把 speaker=SELF 的消息作为行为链证据，但 action/outcome 里不要写 SELF、具体昵称或用户 ID；主体信息只写到 actor_type。
- 不要摘抄原话，不要把一次性的具体任务写成泛化行为。
- source_ids 必须引用支撑这条模式的来源行编号，可以是一条或多条。
- 如果没有合适模式，输出空数组。

只输出 JSON 数组，格式如下：
[
  {{"segment_id":"s1","actor_type":"other_user","learning_type":"observed_behavior","action":"先短句共情，再给一个可执行的小建议","outcome":"对话继续推进，对方更愿意补充细节","source_ids":["2","4","5"]}},
  {{"segment_id":"s2","actor_type":"maibot_self","learning_type":"self_reflection","action":"先承认信息不足，再追问一个关键配置点","outcome":"对方补充配置细节，排查方向变得明确","source_ids":["7","8"]}}
]
""".strip()

BEHAVIOR_FEEDBACK_PROMPT = """
你是 {bot_name} 的行为评估器。

由于先前的聊天场景，你规划预期使用下面的行为进行行动
现在你需要评估这么做之后的结果，包括行动有没有真的被采取，以及行动之后的结果，是否有结果，结果是什么。

被选择的行动参考：
{behavior_references}

评估原则：
- 你需要首先评估有没有按照行动参考执行。先判断“采用证据”，再判断“结果证据”。
- adopted 表示 {bot_name} 的后续真实回复是否实际采用了该行为，必须有 SELF 聊天消息作为证据。
- 如果行动参考包含多个关键步骤，核心步骤缺失时不能给 success；只能给 partial_success，或者证据不足时不输出。
- observation-learning 路径也只有在 {bot_name} 后续真实回复借鉴并采用了其中的核心行为时才算 adopted；不要因为用户或群体自然出现了类似行为就给反馈。
- 只有能从后续聊天中看出较明确结果时才输出反馈；证据不足时不要输出。
- status 只能是 "success"、"partial_success" 或 "failed"。
- success 表示核心行为被完整采用，并且采用后对话推进、用户补充信息、情绪缓和、问题变清楚或结果符合 expected_outcome。
- partial_success 表示只采用了部分关键行为，或结果只是轻微变好/弱匹配 expected_outcome。
- failed 表示采用后明显没有推进、引发反感、偏离需求、造成误解或结果与 expected_outcome 明显相反。
- score_delta 类似奖励预测误差：success 用 0.5 到 1.0；partial_success 用 0.1 到 0.35；failed 用 -0.4 到 -1.0。
- source_ids 引用后续多条 timeline_item 消息中的 item_id，优先引用 chat_message 项，不要引用不存在的 item_id。
- reason 要简短说明采用证据和结果证据。
- outcome 写本次实际观察到的结果，不要复述 expected_outcome。

只输出 JSON 对象：
{{
  "feedback": [
    {{
      "behavior_id": 123,
      "adopted": true,
      "status": "success",
      "score_delta": 0.6,
      "reason": "{bot_name} 采用了先追问关键配置的策略，用户随后补充了配置路径，排查继续推进。",
      "outcome": "用户补充配置路径并继续排查。",
      "source_ids": ["m2", "m3"]
    }}
  ]
}}

如果没有足够确定的反馈，输出：
{{"feedback":[]}}
""".strip()

BEHAVIOR_SCENE_ANALYZE_PROMPT = """
你是 {bot_name} 的行为表现情景分析器。你会看到最近聊天上下文。请把它拆成 1 到 5 个可用于检索、学习和选择行为表现的场景片段。

分析原则：
- 不要续写聊天，不要决定是否回复，也不要替 planner 制定计划。
- 只描述和“应该使用什么行为表现”有关的信息。
- 如果上下文只有一个清晰主题，就只输出 1 个片段；只有在群聊多线程、话题切换或同时存在不同互动目标时，才拆成多个片段。
- 每个片段都要有 source_ids，引用它覆盖的来源行编号；片段之间可以少量重叠，但不要把无关消息硬塞进同一个片段。
- summary 要写成可复用的场景摘要，不要写一次性细节。
- profile 不再直接输出散 tag，而是输出 tag_clusters、need 和 other_traits。
- tag_clusters 只表示场景领域概念，不要包含行为需求。每个 tag_clusters 项只能包含 tag_name、tag_aliases，不要输出 kind。
- need 表示当前需要的行为策略，单独输出为对象，只包含 tag_name、tag_aliases。
- other_traits 表示这个场景中他人或群友的特点、态度、情绪姿态和互动方式，输出为 tag 簇数组；每项只包含 tag_name、tag_aliases，不要输出 kind。
- 每个 tag 簇代表一个单一、可替代匹配的概念；命中 tag_name 或任一 tag_aliases 成员，都应视为命中整个簇。
- tag_clusters 的目标是尽可能找出独立、有意义、可复用的领域概念簇；不是把所有相关词压成少数大簇。
- 每个簇只表示一个概念。相关但不等价的概念要拆成多个 tag_cluster，而不是塞进同一个 tag_aliases。
- tag_name 是这个簇最稳定、最概括的中文短名；tag_aliases 尽量给出 3 到 6 个精确等价、近义、别名、缩写或常见口语表达。
- alias 不能少：它用于提高 tag 簇命中率。除非片段极短或概念本身没有常见别名，否则每个领域簇和 need 都应尽量提供至少 3 个 tag_aliases。
- alias 必须精确：如果没有把握它们是同一个概念的不同措辞，就不要写；不要为了凑数量补相关词。
- tag_aliases 只能表示同一个概念的不同措辞，不能写上位分类、相关领域、使用场景、属性、子功能或并列对象。
- 例如“NapCat / NapCat 框架 / NapCat 配置”可以同簇；“QQ机器人框架 / OneBot / 端口冲突检测”不是同一个概念，必须拆成不同 domain。
- 例如“二次元文化 / ACG文化”可以同簇；“动漫图片 / 猫娘 / 颜文字 / 病娇 / 表情包斗图”不是同一个概念，必须拆成不同 domain。
- 领域 tag_cluster 要主动拆细：每个有效片段至少输出 2 个领域簇，信息足够时输出 3 到 5 个领域簇。只有片段极短、纯寒暄、或确实没有明确领域时，才允许少于 2 个领域簇。
- 多个领域簇应从不同角度描述同一片段，例如具体主题、技术/内容领域、交互对象、应用场景、问题类型、媒介形态或社群文化；不要为了凑数塞入无关主题。
- 不要把同一场景中的并列主题塞进同一个 tag_cluster。比如“AI性能反馈/响应延迟”可以同簇，但“人格化互动”必须另建一个 domain 簇。
- 如果一个领域下同时出现实体、协议/框架、故障类型、操作动作、媒介形态、社群梗或角色属性，它们通常是不同概念，应优先拆成多个领域簇。
- need 通常输出 1 个策略对象；如果存在两种策略，选择当前最主要的一种，不要输出多个 need。
- other_traits 通常输出 1 到 3 个，例如“用户焦急困惑”“群友调侃玩梗”“对方强势质疑”“用户认真求助”。它描述对方状态，不要写麦麦应该怎么做。
- 不要输出 phase、risk、kind、domain_tags、behavior_need、cluster_key、display_name 或其他字段。
- 如果上下文没有明确场景，也要输出 1 个低 confidence 片段。

只输出 JSON 对象：
{{
  "segments": [
    {{
      "segment_id": "s1",
      "title": "简短片段标题",
      "source_ids": ["1", "2", "3"],
      "profile": {{
        "summary": "当前片段的一句话可复用摘要",
        "tag_clusters": [
          {{"tag_name": "模型API配置", "tag_aliases": ["模型接口配置", "模型服务配置", "LLM API 配置", "模型接入参数"]}},
          {{"tag_name": "服务接入故障", "tag_aliases": ["接口连接故障", "服务连接失败", "API 接入失败", "服务调用异常"]}},
          {{"tag_name": "部署运行环境", "tag_aliases": ["部署环境", "运行环境配置", "本地运行环境", "服务运行环境"]}},
          {{"tag_name": "运行日志分析", "tag_aliases": ["查看日志", "日志排查", "分析日志", "日志定位问题"]}}
        ],
        "need": {{"tag_name": "技术排障指导", "tag_aliases": ["排障建议", "故障定位指导", "技术排查建议", "定位问题步骤"]}},
        "other_traits": [
          {{"tag_name": "用户困惑求助", "tag_aliases": ["对方不确定", "用户需要确认", "求助排查", "带着问题询问"]}},
          {{"tag_name": "态度认真配合", "tag_aliases": ["愿意补充信息", "配合排查", "认真描述问题", "接受技术建议"]}}
        ],
        "confidence": 0.82
      }}
    }}
  ]
}}
""".strip()


__all__ = [
    "BEHAVIOR_FEEDBACK_PROMPT",
    "BEHAVIOR_LEARN_PROMPT",
    "BEHAVIOR_SCENE_ANALYZE_PROMPT",
]
