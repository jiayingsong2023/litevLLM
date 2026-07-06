# SPDX-License-Identifier: Apache-2.0
from vllm.entrypoints.openai.api_server import (
    _new_chat_request_id,
    _parse_chat_message_content,
)


def test_openai_chat_image_url_blocks_insert_placeholders() -> None:
    prompt, multi_modal_data = _parse_chat_message_content(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": " file:///tmp/cat.png "},
                    },
                    {"type": "text", "text": "briefly"},
                ],
            }
        ]
    )

    assert prompt == "describe\n<image>\nbriefly"
    assert multi_modal_data == {"image": [{"image": "file:///tmp/cat.png"}]}


def test_openai_chat_multiple_image_url_blocks_keep_order() -> None:
    prompt, multi_modal_data = _parse_chat_message_content(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}},
                    {"type": "text", "text": "with"},
                    {"type": "image_url", "image_url": {"url": "file:///tmp/b.png"}},
                ],
            }
        ]
    )

    assert prompt == "compare\n<image>\nwith\n<image>"
    assert multi_modal_data == {
        "image": [
            {"image": "file:///tmp/a.png"},
            {"image": "file:///tmp/b.png"},
        ]
    }


def test_openai_chat_request_ids_are_unique() -> None:
    assert _new_chat_request_id() != _new_chat_request_id()
