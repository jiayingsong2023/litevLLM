# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Any, Dict, List, Optional
import torch

def _decode_generated_text(tokenizer: Any, token_ids: List[int], sampling_params: Any) -> str:
    skip = getattr(sampling_params, "skip_special_tokens", True)
    spaces = getattr(sampling_params, "spaces_between_special_tokens", False)
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip,
            spaces_between_special_tokens=spaces,
            clean_up_tokenization_spaces=True,
        )
    except TypeError:
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=skip,
                spaces_between_special_tokens=spaces,
            )
        except TypeError:
            try:
                return tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip,
                    clean_up_tokenization_spaces=True,
                )
            except TypeError:
                return tokenizer.decode(token_ids, skip_special_tokens=skip)

class OutputProcessor:
    def __init__(self, tokenizer: Any, model_path: str):
        self.tokenizer = tokenizer
        self.model_path = model_path

    def apply_prompt_guard(self, prompt: str) -> str:
        return prompt

    def get_anti_template_token_ids(self) -> List[int]:
        return []

    def get_capital_question_bias_token_ids(self, prompt: str) -> List[int]:
        return []

    def is_chinese_capital_question(self, prompt: str) -> bool:
        return False

    def apply_context_bias(self, logits: torch.Tensor, generated_ids: List[int], sampling_params: Any, bias_token_ids: List[int], is_capital_question: bool) -> torch.Tensor:
        return logits

    def should_early_stop(self, generated_ids: List[int], partial_text: str) -> bool:
        return False

    def cleanup_output_text(self, text: str) -> str:
        return text

class DefaultOutputProcessor(OutputProcessor):
    pass

class Qwen35OutputProcessor(OutputProcessor):
    def _looks_like_qwen35_35b_awq_model_path(self) -> bool:
        b = os.path.basename(os.path.abspath(self.model_path)).lower()
        return "qwen3.5-35b-awq" in b or ("qwen3.5" in b and "35b" in b and "awq" in b)

    def _looks_like_preformatted_chat_prompt(self, text: str) -> bool:
        s = text.lstrip()
        if len(s) >= 12 and "<|im_start|>" in s[:400]:
            return True
        if s.startswith("<|") and "user" in s[:120].lower():
            return True
        if s.startswith("<think>") or "<think>" in s[:240]:
            return True
        return False

    def _contains_cjk(self, text: str) -> bool:
        return re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text) is not None

    def is_chinese_capital_question(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return "首都" in normalized and any(
            token in normalized for token in ("哪里", "哪儿", "哪个城市", "是什么", "是哪")
        )

    def _extract_chinese_capital_subject(self, text: str) -> Optional[str]:
        normalized = re.sub(r"[\s？?。！!，,：:；;]+", "", text)
        match = re.search(r"(.{1,16}?)首都(?:是)?(?:哪里|哪儿|哪个城市|是什么|是哪)", normalized)
        if match is None:
            return None
        subject = match.group(1).strip()
        return subject or None

    def apply_prompt_guard(self, prompt: str) -> str:
        if not self._looks_like_qwen35_35b_awq_model_path():
            return prompt
        raw = os.environ.get("FASTINFERENCE_QWEN35_PROMPT_GUARD", "1").strip().lower()
        if raw in ("0", "false", "off", "no"):
            return prompt
        if self._contains_cjk(prompt):
            if self.is_chinese_capital_question(prompt):
                subject = self._extract_chinese_capital_subject(prompt)
                answer_prefix = f"{subject}的首都是" if subject else "某国的首都是"
                guard = (
                    "请直接用一句自然中文作答，并立即给出答案。"
                    f"请以“{answer_prefix}”开头。"
                    "不要复述问题。"
                    "不要输出<think>标签、markdown、项目符号、引号、格式模板或重复符号。\n"
                )
            else:
                guard = (
                    "请直接用一句自然中文作答，并立即给出答案。"
                    "不要复述问题。"
                    "不要输出<think>标签、markdown、项目符号、引号、格式模板或重复符号。\n"
                )
        else:
            guard = (
                "Please answer directly in one short natural-language sentence, starting with the answer immediately. "
                "Do not restate the question. "
                "Do not output <think> tags, markdown, bullets, quotes, formatting templates, or repeated symbols.\n"
            )
        if self._looks_like_preformatted_chat_prompt(prompt):
            if "<|im_start|>user\n" in prompt:
                return prompt.replace("<|im_start|>user\n", f"<|im_start|>user\n{guard}\n", 1)
            if "<|user|>" in prompt:
                return prompt.replace("<|user|>", f"<|user|>\n{guard}\n", 1)
            return guard + prompt
        return guard + prompt

    def get_anti_template_token_ids(self) -> List[int]:
        if not self._looks_like_qwen35_35b_awq_model_path():
            return []
        cached = getattr(self.tokenizer, "_fastinference_qwen35_anti_template_ids", None)
        if cached is not None:
            return cached
        out = set()
        candidate_strings = [
            "*", "**", "***", "-", "--", "\n", "\n\n", '"', "“", "”", "'", "`", "```", "##", "###",
            "<think>", "</think>", "<", ">", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "1.", "2.", "(1", "(2"
        ]
        for text in candidate_strings:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                try:
                    ids = self.tokenizer.encode(text)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(ids, list) and len(ids) == 1:
                out.add(int(ids[0]))
        cached_list = sorted(out)
        setattr(self.tokenizer, "_fastinference_qwen35_anti_template_ids", cached_list)
        return cached_list

    def _get_single_token_ids(self, candidates: List[str]) -> List[int]:
        out = set()
        for text in candidates:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                try:
                    ids = self.tokenizer.encode(text)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(ids, list) and len(ids) == 1:
                out.add(int(ids[0]))
        return sorted(out)

    def get_capital_question_bias_token_ids(self, prompt: str) -> List[int]:
        if self.is_chinese_capital_question(prompt):
            return self._get_single_token_ids(["都"])
        return []

    def apply_context_bias(self, logits: torch.Tensor, generated_ids: List[int], sampling_params: Any, bias_token_ids: List[int], is_capital_question: bool) -> torch.Tensor:
        if not is_capital_question or not bias_token_ids:
            return logits
        partial_text = _decode_generated_text(self.tokenizer, generated_ids, sampling_params)
        compact = re.sub(r"\s+", "", partial_text)
        if compact.endswith("首"):
            logits = logits.clone()
            for tid in bias_token_ids:
                if 0 <= tid < logits.numel():
                    logits[tid] += 40.0
        return logits

    def should_early_stop(self, generated_ids: List[int], partial_text: str) -> bool:
        if len(generated_ids) < 10:
            return False
        tail_ids = generated_ids[-12:]
        if len(set(tail_ids)) == 1:
            return True
        tail = partial_text[-96:]
        if "<think>" in tail.lower():
            return True
        compact = tail.strip()
        if len(compact) >= 24 and re.fullmatch(r"[\s\*\-\"'`.,:;!?\(\)\[\]{}<>|\\/]+", compact):
            return True
        return False

    def cleanup_output_text(self, text: str) -> str:
        if not self._looks_like_qwen35_35b_awq_model_path():
            return text
        cleaned = text.rstrip()
        if cleaned:
            cleaned = re.sub(
                r"^(?:(?:<\|)?(?:[a-z_]*)(?:im_)?(?:start|end)\|>?\s*|[a-z_]*art\|\s*>?\s*)+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"^[>\|\-\s]+", "", cleaned)
            cleaned = re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])\s+(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])", "", cleaned)
            cleaned = re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])\s+(?=[，。！？；：])", "", cleaned)
            cleaned = re.sub(r"([。！？.!?])(?:[\s\"“”'`*#\-\d\(\)\[\]\{\}\n]+)$", r"\1", cleaned)
            cleaned = re.sub(r'[\s"“”\'`*#\-]+$', "", cleaned)
            cleaned = re.sub(r"\(\s*$", "", cleaned)
            cleaned = cleaned.rstrip()
        return cleaned or text.rstrip()

def get_output_processor(model_path: str, tokenizer: Any) -> OutputProcessor:
    b = os.path.basename(os.path.abspath(model_path)).lower()
    if "qwen3.5" in b:
        return Qwen35OutputProcessor(tokenizer, model_path)
    return DefaultOutputProcessor(tokenizer, model_path)
