EXPRESSION_LEARN_PROMPT = """
{chat_str}
你需要完成一个提取任务
任务1：请从上面这段群聊中用户的语言风格和说话方式
1. 只考虑文字，不要考虑表情包和图片
2. 不要总结SELF的发言，因为这是你自己的发言，不要重复学习你自己的发言
3. 不要涉及具体的人名，也不要涉及具体名词
4. 思考有没有特殊的梗，一并总结成语言风格
5. 例子仅供参考，请严格根据群聊内容总结!!
注意：总结成如下格式的规律，总结的内容要详细，但具有概括性：
例如：当"AAAAA"时，可以"BBBBB", AAAAA代表某个场景，不超过20个字。BBBBB代表对应的语言风格，特定句式或表达方式，不超过20个字。
表达方式在3-5个左右，不要超过10个


输出要求：
将表达方式，语言风格以 JSON 数组输出，每个元素为一个对象，结构如下（注意字段名）：
注意请不要输出重复内容，请对表达方式进行去重。

[
  {{"situation": "AAAAA", "style": "BBBBB", "source_id": "3"}},
  {{"situation": "CCCC", "style": "DDDD", "source_id": "7"}}
  {{"situation": "对某件事表示十分惊叹", "style": "使用 我嘞个xxxx", "source_id": "3"}},
  {{"situation": "表示讽刺的赞同，不讲道理", "style": "对对对", "source_id": "4"}},
  {{"situation": "当涉及游戏相关时，夸赞，略带戏谑意味", "style": "使用 这么强！", "source_id": "5"}}
]

其中：
表达方式条目：
- situation：表示“在什么情境下”的简短概括（不超过20个字）
- style：表示对应的语言风格或常用表达（不超过20个字）
- source_id：该表达方式对应的来源编号，即上方聊天记录中的 source_id 数字（例如 [source_id:3] 对应 "3"），请只输出数字本身

输出 JSON：
""".strip()

EXPRESSION_EVALUATION_PROMPT = """
请评估以下表达方式或语言风格以及使用条件或使用情景是否合适：
使用条件或使用情景：{situation}
表达方式或言语风格：{style}

请从以下方面进行评估：
{criteria_list}

请以JSON格式输出评估结果：
{{
    "suitable": true/false,
    "reason": "评估理由（如果不合适，请说明原因）"

}}
请严格按照JSON格式输出，不要包含其他内容。
""".strip()


def build_expression_selector_prompt(
    *,
    history_block: str,
    target_text: str,
    reply_reason: str,
    candidate_lines: list[str],
) -> str:
    return (
        "你是聊天机器人的表达方式选择子代理。\n"
        "你只负责根据最近聊天上下文，为这一次可见回复挑选最合适的表达方式。\n"
        "请只从下面候选中选择 0 到 3 条最适合当前语境的表达方式。\n"
        "优先考虑自然、贴合上下文、不生硬、不模板化。\n"
        "如果没有明显合适的，就返回空数组。\n"
        '严格只输出 JSON，对象格式为 {"selected_ids":[123,456]}。\n\n'
        f"最近上下文：\n{history_block}\n\n"
        f"目标消息：{target_text or '无'}\n"
        f"回复理由：{reply_reason.strip() or '无'}\n\n"
        f"候选表达方式：\n{chr(10).join(candidate_lines)}"
    )


__all__ = [
    "EXPRESSION_EVALUATION_PROMPT",
    "EXPRESSION_LEARN_PROMPT",
    "build_expression_selector_prompt",
]
