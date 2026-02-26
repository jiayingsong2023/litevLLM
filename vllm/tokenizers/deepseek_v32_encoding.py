# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# copy from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
import copy
import json
from typing import Any

import regex as re

# flake8: noqa: E501

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"
system_msg_template: str = "{content}"
user_msg_template: str = "<｜User｜>{content}<｜Assistant｜>"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"
thinking_template = "{reasoning}"

response_format_template: str = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
tool_call_template: str = (
    '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
)
tool_calls_template = (
    "<{dsml_token}function_calls>\n{tool_calls}\n</{dsml_token}function_calls>"
)

tool_output_template: str = "\n<result>{content}</result>"

def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)

def tools_from_openai_format(tools):
    return [tool["function"] for tool in tools]

def tool_calls_from_openai_format(tool_calls):
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]

def tool_calls_to_openai_format(tool_calls):
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        for tool_call in tool_calls
    ]

def encode_arguments_to_dsml(tool_call: dict[str, str]) -> str:
