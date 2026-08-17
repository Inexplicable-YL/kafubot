# kafubot

一个直接服务于拟人社交智能体的 QQ/OneBot 运行时。项目不再依赖 SekaiBot，
也不再发现或执行 Node；当前协议适配器只实现 CQHTTP/OneBot 11，但协议边界仍然
保留，便于以后增加其他适配器。

## Agent Loop

群聊和私聊只是不同的上下文分区，不对应不同的处理器或 Agent。系统中始终只有
一个人格、一个 Social Environment 和一个 Main Executive：

```text
CQHTTP Adapter（解析完整 OneBot 11 事件族）
        │
        ▼
Bot 类型过滤（仅 GroupMessageEvent / PrivateMessageEvent）
        │
        ▼  媒体预处理和环境写入
Social Environment ──► Global Gate（只决定是否唤醒）
        │                         │
        │                         ▼
        └──────────────► Social Home + Attention Scheduler
                                  │
                                  ▼
                         one Main Executive
                       open_chat / read_more
                  inspect_person / inspect_media
                    search_memory / skip / reply
                                  │
                         Action Contract
                                  ▼
                    Context Compiler ──► Replyer
                                  │
                                  ▼
                         QQActions / CQHTTP
```

关键不变量：

- 事件到达只更新世界并通知 Gate，不会直接执行 `ainvoke(session)`。
- CQHTTP 保留完整事件能力，但主循环入口只过滤并接收群消息、私聊消息；不存在
  额外的“标准事件”转换层。
- Executive 初始只看压缩后的 Social Home，必须通过工具渐进展开局部会话。
- 同一轮可以处理多个会话，但受 Focus Budget、最大步骤数和回复数限制。
- `reply` 先产生 Action Contract；Replyer 只负责措辞，无权重新选择目标或行动。
- `SocialAgentRuntime` 持有并运行 `agent.builder.create_agent` 图，同时管理每轮的
  `SocialAgentContext`；默认的 `ExecutiveMiddleware` 只提供核心提示和基础工具，
  可以被其他 Middleware 替换。`middleware=None` 使用默认插件，显式传入的
  `middleware` 序列则作为完整插件栈。当前没有接入旧 Middleware。
- 持久化内容仅包括 Self State、Active Threads、Recent Actions 和 State Delta，
  不保存模型思维链或 ReAct scratchpad。
- 当前只接入会话世界模型，不加载旧 Middleware；`WorldModelProvider` 仅保留为后续
  能力接入边界。

核心实现位于 [`kafubot/agency`](kafubot/agency)，其中：

- `environment.py`：统一会话时间线、未处理游标和历史仓库。
- `gate.py`：低成本全局唤醒判断。
- `attention.py`：跨会话注意力评分和 Social Home 排序。
- `social.py`：`create_agent` 图宿主、运行上下文、round 与插件生命周期。
- `executive.py`：可替换的默认 Executive Middleware 和渐进披露基础工具。
- `compiler.py`：校验 Action Contract 并编译最小局部上下文。
- `replyer.py`：把语义行动实现为 QQ 消息。
- `providers.py`：世界模型能力的统一读写协议；当前仅有会话上下文 Provider。
- `state.py`：不含推理痕迹的原子状态持久化。

旧 `agent/manager.py` 及其中间件组合不再处于 Bot 的运行调用链中；它们暂时保留，
用于后续把长期记忆、行为学习、表达学习等能力逐项迁移为 World Model Provider。

## 协议与启动

通用适配器系统位于 `kafubot.adapters.Adapter`，轮询、HTTP 与 WebSocket 可继承
基类位于 `kafubot.adapters.utils`，CQHTTP 实现在 `kafubot.adapters.cqhttp`。
新适配器可通过 `register_adapter()` 注册，并用自身的 `Config` 类绑定
`[adapter.<name>]` 配置。当前配置使用 `[adapter.cqhttp]` 和统一的 `[agent]`，
主循环参数分别位于 `[agent.gate]`、`[agent.attention]`、
`[agent.executive]`。

```powershell
uv run python main.py
```

运行验证：

```powershell
uv run ruff check agent kafubot main.py tests
uv run pytest -q
uv run pyright agent kafubot tests
```
