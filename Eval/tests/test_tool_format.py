from openo3_eval.video_o3.tool_format import rewrite_messages_for_tool_format


def test_rewrite_messages_for_tool_format_rewrites_user_list_text_items() -> None:
    messages = [
        {"role": "system", "content": 'Use <tool_call>{"name": "crop_video", "arguments": {"start_time": START_TIMESTAMP, "end_time": END_TIMESTAMP, "sampling_strategy": "medium"}}</tool_call>'},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Example: <tool_call>{"name": "crop_video", "arguments": {"start_time": START_TIMESTAMP, "end_time": END_TIMESTAMP, "sampling_strategy": "medium"}}</tool_call>',
                },
                {"type": "image", "image": "fixtures/fake.jpg"},
            ],
        },
    ]

    rewritten = rewrite_messages_for_tool_format(messages, tool_format="qwen3_5")

    print(rewritten)

    assert "<function=crop_video>" in rewritten[0]["content"]
    assert "<function=crop_video>" in rewritten[1]["content"][0]["text"]
    assert '"name": "crop_video"' not in rewritten[1]["content"][0]["text"]

test_rewrite_messages_for_tool_format_rewrites_user_list_text_items()
