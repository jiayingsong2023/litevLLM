#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.async_llm import AsyncLLM
from vllm.serving.config_builder import build_vllm_config


@dataclass(frozen=True)
class SpotcheckCase:
    case_id: str
    prompt: str
    keywords: tuple[str, ...]
    image_name: str


DEFAULT_CASES: tuple[SpotcheckCase, ...] = (
    SpotcheckCase(
        case_id="red_square",
        prompt="What color is the square? Answer with one word.",
        keywords=("red",),
        image_name="red_square.png",
    ),
    SpotcheckCase(
        case_id="blue_object",
        prompt="What color is the object? Answer with one word.",
        keywords=("blue",),
        image_name="blue_circle.png",
    ),
)


def create_fixture_images(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 120, 120), fill="red")
    image.save(directory / "red_square.png")

    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 120, 120), fill="blue")
    image.save(directory / "blue_circle.png")

    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 62), "HI", fill="black")
    image.save(directory / "ocr_hi.png")


def score_output(text: str, keywords: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(keyword.casefold() in folded for keyword in keywords)


async def run_case(
    llm: AsyncLLM,
    case: SpotcheckCase,
    image_dir: Path,
    max_tokens: int,
    prompt: str,
) -> tuple[bool, str, list[int]]:
    output = None
    async for item in llm.generate(
        prompt,
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        request_id=f"gemma4-mm-quality-{case.case_id}",
        multi_modal_data={
            "image": [{"image": f"file://{image_dir / case.image_name}"}],
        },
    ):
        output = item
    if output is None or not output.outputs:
        return False, "", []
    text = str(output.outputs[0].text)
    token_ids = [int(t) for t in (output.outputs[0].token_ids or [])]
    return score_output(text, case.keywords), text, token_ids


def apply_multimodal_chat_template(tokenizer, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


async def run_spotcheck(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="fastinference-gemma4-mm-quality-") as tmp:
        image_dir = Path(tmp)
        create_fixture_images(image_dir)
        cfg = build_vllm_config(
            args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = AsyncLLM.from_vllm_config(cfg)
        cases = DEFAULT_CASES[: args.limit]
        passed = 0
        try:
            for case in cases:
                prompt = apply_multimodal_chat_template(tokenizer, case.prompt)
                ok, text, token_ids = await run_case(
                    llm,
                    case,
                    image_dir,
                    args.max_tokens,
                    prompt,
                )
                passed += int(ok)
                status = "PASS" if ok else "FAIL"
                print(
                    f"{status} {case.case_id}: expected={case.keywords} "
                    f"text={text!r} tokens={token_ids}"
                )
        finally:
            llm.shutdown()

    total = len(cases)
    print(f"SUMMARY pass={passed} total={total}")
    return 0 if passed == total else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma4 multimodal semantic spotcheck with synthetic images",
    )
    parser.add_argument(
        "--model",
        default="models/gemma-4-31B-it-AWQ-4bit",
        help="Gemma4 model path",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--limit",
        type=int,
        default=len(DEFAULT_CASES),
        help="number of built-in cases to run",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_spotcheck(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
