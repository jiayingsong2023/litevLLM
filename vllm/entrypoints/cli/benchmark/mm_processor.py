# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.mm_processor import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase

class BenchmarkMMProcessorSubcommand(BenchmarkSubcommandBase):
