# kafubot

QQ/OneBot 群聊与私聊 Agent。内部会话结构和社交动作都是推断层；可见输出只编译为当前适配器实际支持的文本、单条引用回复、真实 QQ 号 `@` 与图片消息段。

## 社交控制评测

- `uv run python scripts/social_eval.py`：汇总介入率、引用率、行动目标、纠错/负反馈、目标续聊率和效果归因置信度。
- `uv run python scripts/conversation_replay.py group_群号`：从 `.database/qq_event_ledger.db` 重放真实 QQ 可观测事件，检查多话题、显式/推断指代及群体状态。

事件账本只保存消息 ID、会话、发送者、时间、文本化消息段、reply/@/to_me 与图片存在标记，不保存图片 base64。`/clear` 或 `/清除` 会清除当前 QQ 会话的历史和派生状态；已经形成的通用行为/表达经验不会被当作聊天历史误删。
