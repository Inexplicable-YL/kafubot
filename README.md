# kafubot

一个面向拟人社交智能体的 QQ/OneBot 运行时。协议事件只负责更新可观察世界，
所有会话共享一个 Main Executive；会话是上下文分区，而不是独立人格或独立 Agent。

## 架构

```text
CQHTTP / OneBot 11
        │
        ▼
Bot：有界事件队列、Worker、生命周期
        │
        ▼
PluginHost：发现、依赖、配置、所有权、回滚、生命周期
        │
        ├── Ingress ──► Environment ──► Observation Hooks
        ├── Gate / Attention / Self State / World Providers
        ├── Tool(HOME/OPEN) / Middleware / Reply Hooks
        └── Model / Executive / Replyer
                         │
                         ▼
               HOME ⇄ OPEN:<session_id> ──► QQ
```

[`kafubot/social.py`](kafubot/social.py) 只承载协议事件与插件图之间的稳定运行时循环。
Environment、Global Gate、Attention、World Provider、主模型、Main Executive、
Replyer 和 Ingress 均由插件提供；运行时不会在插件关闭后偷偷创建备用组件。
项目不再保留旧 `agent/` 入口，也没有 classic/executive 两套执行路径。

Main Executive 初始只看到跨会话压缩后的 Social Home。它必须先用 `open_chat`
打开一个会话，再按需读取历史、人物、媒体、记忆或插件工具；最终以 `reply` 提交
Action Contract，或以 `quit` 明确保持沉默。Replyer 只实现措辞，不重新决定目标、
立场或是否发送表情包。

## 万物皆插件

插件系统位于 [`kafubot/cognition/plugins`](kafubot/cognition/plugins)：

- 文件就是功能边界：单文件插件必须在该文件导出
  `plugin = PluginDefinition(...)`；复杂插件使用一个包，并且只在包的
  `__init__.py` 导出 `plugin`。Loader 只扫描插件目录的直接子文件/包，未导出
  `plugin` 的 `base.py`、`loader.py` 等基础设施不会被视为插件，也不存在
  `builtin/` 或集中插件清单。
- `PluginContext` 是唯一扩展面：同一 API 注册 Service/Resource、Tool、Middleware、
  World Provider、启动完成回调、会话清理和所有运行时 Hook。不存在
  `SocialPluginHub`、兼容 Bridge 或第二套插件生命周期。
- `PluginHost` 执行依赖拓扑、配置预校验和两阶段装配。`apply` 阶段发布能力，
  `ready` 阶段在全图可见后组合 Executive；失败时按插件和资源的相反顺序回滚。
- Tool 可声明 `HOME`、`OPEN` 或 `BOTH` 作用域。所有工具只向
  执行图注册一次，Executive 每轮仅暴露当前状态的子集，并在实际调用前再次校验
  状态，因此作用域不依赖提示词约束。
- `on_observe`、`on_peek`、`on_query`、`on_prepare_reply`、`on_guard_reply`、
  `on_reply_committed` 和 `on_skip_committed` 直接注册到 Host。事件 API 提供当前
  原生消息，不再重建旧 `ManagerState` 或伪造 `ToolRuntime`。
- 观察、知识补充和提交后处理采用故障隔离；回复守卫采用失败关闭，避免限流或
  安全组件异常时继续发送消息。

一个插件可以同时贡献 Tool、Middleware 与生命周期行为：

```python
from kafubot.cognition.plugins import PluginContext, PluginDefinition, ToolScope


def apply(context: PluginContext, config: dict[str, object]) -> None:
    service = context.resource("my_service", MyService(**config))
    context.tool(service.search_tool, ToolScope.OPEN)
    context.middleware(service.middleware)
    context.on_observe(service.observe)
    context.on_prepare_reply(service.prepare_reply)
    context.clear_session(service.clear_session)


plugin = PluginDefinition(name="my_plugin", apply=apply)
```

第三方包只需暴露 entry point，无需修改本仓库：

```toml
[project.entry-points."kafubot.plugins"]
my_plugin = "my_package.kafubot_plugin"
```

当前组件均由 `[agent.plugins.<name>]` 独立控制：

| 插件 | 接入当前运行时的职责 |
| --- | --- |
| `environment` | 提供跨会话可观察环境；可由测试或部署注入实现 |
| `self_state` | 提供 Focus、Active Threads、Fatigue 等自身状态存储 |
| `model` | 提供 Main Executive 主模型工厂 |
| `ingress` | 将 OneBot 文本、引用和图片转成稳定认知消息 |
| `attention` | 从所有活跃会话构造 Social Home |
| `world_model` | 聚合 Conversation Provider、插件 Provider 与查询 Hook |
| `replyer` | 根据 Action Contract 生成并发送可见回复 |
| `interaction` | 启用 Main Executive 的 HOME/OPEN 交互中间件 |
| `social_signals` | 可复用的社交信号分析服务 |
| `conversation` | 观察会话结构，并向回复注入当前会话框架 |
| `reply_effects` | 跟踪发送后的可观察反馈，回写行为和表达效果 |
| `limiter` | 在发送前限流，在可见回复后记账 |
| `time_gate` | 用 Global Gate 决定何时唤醒；关闭后新消息直接唤醒 |
| `meme` | 在 OPEN 提供 `search_meme`，并按 Action Contract 的 `meme_intent` 发送 |
| `jargon` | 后台学习群体黑话，并在相关回复前提供释义 |
| `memory` | 后台写入长期记忆，并提供自动召回和 `search_memory` 查询 |
| `summarization` | 后台维护会话摘要，并在回复前提供摘要 |
| `behavior` | 准备行为参考、学习行为模式并接收可观察效果反馈 |
| `expression` | 准备表达习惯、学习表达模式并接收可观察效果反馈 |
| `prompt` | 启用 Main Executive 系统提示 |
| `clock` | 向 Executive 暴露显示时区中的当前时间 |
| `search_song` | 在 OPEN 状态提供歌曲检索工具 |
| `view_message` | 在 OPEN 状态读取当前 QQ 会话中的合并转发消息 |

`conversation` 和 `reply_effects` 依赖 `social_signals`；`interaction` 依赖
Environment、Model、Replyer、Self State 和 World Model。启用依赖方却关闭依赖、
配置未知插件或形成循环依赖时，启动会直接报错；已创建的数据库、后台任务、
Provider、HTTP Client 和 SQLAlchemy Engine 会被回滚释放。

示例：

```toml
[agent.plugins.memory]
enabled = true

[agent.plugins.meme]
enabled = false

```

旧实现里依赖另一套 Manager Agent、或已被当前多模态摄取和 Main Executive 覆盖的
`query_image`、`query_expert`、调试工具注入及延迟工具中间件没有保留。歌曲检索和
转发消息读取则改造成只在当前 OPEN 会话中可用的插件工具。

## 目录职责

- [`kafubot/agency`](kafubot/agency)：不依赖插件生命周期的 Action Contract、
  Social Home 和 Self State 数据模型。
- [`kafubot/cognition/plugins`](kafubot/cognition/plugins)：插件宿主，以及完整拥有
  Environment、Gate、Attention、Ingress、Executive、Replyer、Provider、Tool 和
  学习逻辑的单文件/包插件。
- [`kafubot/cognition`](kafubot/cognition)：插件之外可复用的认知消息、模型、媒体
  与遥测基础设施。
- [`kafubot/runtimes`](kafubot/runtimes)：对 Bot 暴露唯一的运行时协议与工厂。
- [`kafubot/adapters`](kafubot/adapters)：协议适配器；当前实现 CQHTTP/OneBot 11。

Self State 仅持久化 Focus、Active Threads、Recent Actions、Fatigue 和 State Delta，
不保存模型思维链或 ReAct scratchpad。

## 启动与验证

```powershell
uv run python main.py
```

```powershell
uv run ruff check kafubot main.py scripts tests
uv run ruff format --check kafubot main.py scripts tests
uv run pytest -q
uv run pyright
```
