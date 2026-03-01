# SPDX-License-Identifier: Apache-2.0
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class AsyncEngineArgs:
    model: str = ""
    
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--model", type=str, default="default_model")
        return parser
    
    def create_engine_config(self, **kwargs):
        return None

    @classmethod
    def from_cli_args(cls, args):
        return cls(model=args.model)
