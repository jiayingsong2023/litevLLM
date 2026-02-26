# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os

from huggingface_hub import HfApi, snapshot_download

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry
from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver

logger = init_logger(__name__)

class HfHubResolver(FilesystemResolver):
    def __init__(self, repo_list: list[str]):
        logger.warning(
            "LoRA is allowing resolution from the following repositories on"
            " HF Hub: %s please note that allowing remote downloads"
            " is not secure, and that this plugin is not intended for use in"
            " production environments.",
            repo_list,
        )

        self.repo_list: list[str] = repo_list
        self.adapter_dirs: dict[str, set[str]] = {}

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        # If a LoRA name begins with the repository name, it's disambiguated
        maybe_repo = await self._resolve_repo(lora_name)

        # If we haven't inspected this repo before, save available adapter dirs
        if maybe_repo is not None and maybe_repo not in self.adapter_dirs:
            self.adapter_dirs[maybe_repo] = await self._get_adapter_dirs(maybe_repo)

        maybe_subpath = await self._resolve_repo_subpath(lora_name, maybe_repo)

        if maybe_repo is None or maybe_subpath is None:
            return None

        repo_path = await asyncio.to_thread(
            snapshot_download,
            repo_id=maybe_repo,
            allow_patterns=f"{maybe_subpath}/*" if maybe_subpath != "." else "*",
        )

        lora_path = os.path.join(repo_path, maybe_subpath)
        maybe_lora_request = await self._get_lora_req_from_path(
            lora_name, lora_path, base_model_name
        )
        return maybe_lora_request

    async def _resolve_repo(self, lora_name: str) -> str | None:
        for potential_repo in self.repo_list:
            if lora_name.startswith(potential_repo) and (
                len(lora_name) == len(potential_repo)
                or lora_name[len(potential_repo)] == "/"
            ):
                return potential_repo
        return None

    async def _resolve_repo_subpath(
        self, lora_name: str, maybe_repo: str | None
    ) -> str | None:
        if maybe_repo is None:
            return None
        repo_len = len(maybe_repo)
        if lora_name == maybe_repo or (
            len(lora_name) == repo_len + 1 and lora_name[-1] == "/"
        ):
            # Resolves to the root of the directory
            adapter_dir = "."
        else:
            # It's a subpath; removing trailing slashes if there are any
            adapter_dir = lora_name[repo_len + 1 :].rstrip("/")

        # Only download if the directory actually contains an adapter
        is_adapter = adapter_dir in self.adapter_dirs[maybe_repo]
        return adapter_dir if is_adapter else None

    async def _get_adapter_dirs(self, repo_name: str) -> set[str]:
        repo_files = await asyncio.to_thread(HfApi().list_repo_files, repo_id=repo_name)
        adapter_dirs = {
            os.path.dirname(name)
            for name in repo_files
            if name.endswith("adapter_config.json")
        }
        if "adapter_config.json" in repo_files:
            adapter_dirs.add(".")
        return adapter_dirs

def register_hf_hub_resolver():
