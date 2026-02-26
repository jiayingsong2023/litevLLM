# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.entrypoints.cli.types import CLISubcommand

class BenchmarkSubcommandBase(CLISubcommand):
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError
