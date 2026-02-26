# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Literal

from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import config
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    validate_chat_template,
)
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

class LoRAParserAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ):
        if values is None:
            values = []
        if isinstance(values, str):
            raise TypeError("Expected values to be a list")

        lora_list: list[LoRAModulePath] = []
        for item in values:
            if item in [None, ""]:  # Skip if item is None or empty string
                continue
            if "=" in item and "," not in item:  # Old format: name=path
                name, path = item.split("=")
                lora_list.append(LoRAModulePath(name, path))
            else:  # Assume JSON format
                try:
                    lora_dict = json.loads(item)
                    lora = LoRAModulePath(**lora_dict)
                    lora_list.append(lora)
                except json.JSONDecodeError:
                    parser.error(f"Invalid JSON format for --lora-modules: {item}")
                except TypeError as e:
                    parser.error(
                        f"Invalid fields for --lora-modules: {item} - {str(e)}"
                    )
        setattr(namespace, self.dest, lora_list)

@config
@dataclass
class FrontendArgs:
    port: int = 8000
    uvicorn_log_level: Literal[
        "critical", "error", "warning", "info", "debug", "trace"
    ] = "info"
    disable_access_log_for_endpoints: str | None = None
    allow_credentials: bool = False
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    api_key: list[str] | None = None
    lora_modules: list[LoRAModulePath] | None = None
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    trust_request_chat_template: bool = False
    default_chat_template_kwargs: dict[str, Any] | None = None
    response_role: str = "assistant"
    ssl_certfile: str | None = None
    enable_ssl_refresh: bool = False
    ssl_ciphers: str | None = None
    root_path: str | None = None
    --middleware arguments. The value should be an import path. If a function
    is provided, vLLM will add it to the server using
    `@app.middleware('http')`. If a class is provided, vLLM will
    strings of the form 'token_id:{token_id}' so that tokens that are not
    enable_auto_tool_choice: bool = False
    exclude_tools_when_tool_choice_none: bool = False
    tool_call_parser: str | None = None
    tool_parser_plugin: str = ""
    tool_server: str | None = None
    log_config_file: str | None = envs.VLLM_LOGGING_CONFIG_PATH
    enable_prompt_tokens_details: bool = False
    enable_force_include_usage: bool = False
    --enable-log-outputs is set.
    tokens_only: bool = False
    enable_offline_docs: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        from vllm.engine.arg_utils import get_kwargs

        frontend_kwargs = get_kwargs(FrontendArgs)

        # Special case: allowed_origins, allowed_methods, allowed_headers all
        # need json.loads type
        # Should also remove nargs
        frontend_kwargs["allowed_origins"]["type"] = json.loads
        frontend_kwargs["allowed_methods"]["type"] = json.loads
        frontend_kwargs["allowed_headers"]["type"] = json.loads
        del frontend_kwargs["allowed_origins"]["nargs"]
        del frontend_kwargs["allowed_methods"]["nargs"]
        del frontend_kwargs["allowed_headers"]["nargs"]

        # Special case: default_chat_template_kwargs needs json.loads type
        frontend_kwargs["default_chat_template_kwargs"]["type"] = json.loads

        # Special case: LoRA modules need custom parser action and
        # optional_type(str)
        frontend_kwargs["lora_modules"]["type"] = optional_type(str)
        frontend_kwargs["lora_modules"]["action"] = LoRAParserAction

        # Special case: Middleware needs to append action
        frontend_kwargs["middleware"]["action"] = "append"
        frontend_kwargs["middleware"]["type"] = str
        if "nargs" in frontend_kwargs["middleware"]:
            del frontend_kwargs["middleware"]["nargs"]
        frontend_kwargs["middleware"]["default"] = []

        # Special case: disable_access_log_for_endpoints is a single
        # comma-separated string, not a list
        if "nargs" in frontend_kwargs["disable_access_log_for_endpoints"]:
            del frontend_kwargs["disable_access_log_for_endpoints"]["nargs"]

        # Special case: Tool call parser shows built-in options.
        valid_tool_parsers = list(ToolParserManager.list_registered())
        parsers_str = ",".join(valid_tool_parsers)
        frontend_kwargs["tool_call_parser"]["metavar"] = (
            f"{{{parsers_str}}} or name registered in --tool-parser-plugin"
        )

        frontend_group = parser.add_argument_group(
            title="Frontend",
            description=FrontendArgs.__doc__,
        )

        for key, value in frontend_kwargs.items():
            frontend_group.add_argument(f"--{key.replace('_', '-')}", **value)

        return parser

def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=None,
        help="How many API server processes to run. "
        "Defaults to data_parallel_size if not specified.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html",
    )
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser

def validate_parsed_serve_args(args: argparse.Namespace):
