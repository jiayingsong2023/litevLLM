# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_torch_equal

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

# ===================================================
# torch 2.9 Inductor PythonWrapperCodegen monkeypatch
# ===================================================
# This change monkeypatches memory_plan_reuse in pytorch 2.9.0 to work around
# a test failure for test_multi_graph_piecewise_compile_outputs_equal.
# For more context, see https://github.com/pytorch/pytorch/pull/165514.

def memory_plan_reuse_patched(self):
    import torch._inductor.ir as ir
    from torch._inductor.codegen.wrapper import (
        EnterSubgraphLine,
        ExitSubgraphLine,
        MemoryPlanningLine,
        MemoryPlanningState,
        SubgraphPythonWrapperCodegen,
    )
    from torch._inductor.virtualized import V

    def get_output_names(graph_outputs) -> list[str]:
        import itertools

        names = []
        shape_counter = itertools.count(0)
        none_counter = itertools.count(0)
        for node in graph_outputs:
            if isinstance(node, ir.NoneAsConstantBuffer):
                names.append(f"{V.graph.name}_none{next(none_counter)}")
            elif isinstance(node, ir.ShapeAsConstantBuffer):
                names.append(f"{V.graph.name}_shape{next(shape_counter)}")
            else:
                names.append(node.get_name())
        return names

    if (
        isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen)
        and V.graph.wrapper_code.partition_signatures is not None
    ):
        out_names = get_output_names(
            V.graph.wrapper_code.partition_signatures.output_nodes
        )
    else:
        out_names = V.graph.get_output_names()

    while (
        self.lines
        and isinstance(self.lines[-1], MemoryPlanningLine)
        and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]
    ):
        # these lines will be pointless
        self.lines.pop()

    # codegen allocations in two passes
    planning_states = [MemoryPlanningState()]
    past_planning_states = []
    for i in range(len(self.lines)):
        line = self.lines[i]
        if isinstance(line, MemoryPlanningLine):
            self.lines[i] = line.plan(planning_states[-1])
        elif isinstance(line, EnterSubgraphLine):
            planning_states.append(MemoryPlanningState())
        elif isinstance(line, ExitSubgraphLine):
            past_planning_states.append(planning_states.pop())
    past_planning_states.append(planning_states.pop())
    assert len(planning_states) == 0

# ===================================================
# torch 2.9 Inductor get_graph_partition_signature monkeypatch
# ===================================================
# This change monkeypatches get_graph_partition_signature in pytorch 2.9.0 to
# fix inductor partition + attention-nvfp4 quant fusion, tested in
# `tests/compile/test_fusion_attn.py::test_attn_quant`.
# For more context, see https://github.com/pytorch/pytorch/pull/165815.

def get_graph_partition_signature_patched(
    self, partitions, skip_cudagraphs: list[bool]
):
    from torch._inductor import dependencies
    from torch._inductor.ir import GraphPartitionSignature, MutationOutput, NoneLayout
    from torch._inductor.virtualized import V
    from torch.utils._ordered_set import OrderedSet

    signatures = []

    unmet_output_names = OrderedSet(V.graph.get_output_names())
    name_to_node = self.get_name_to_nodes()

    def is_none_layout(buf_name: str) -> bool:
        buf = self.name_to_buf.get(buf_name, None)

        if buf is None:
            return False

        if isinstance(buf.node.layout, NoneLayout):
            if isinstance(buf.node, MutationOutput) and (
                real_name := self.mutation_real_name.get(buf_name, None)
            ):
                return is_none_layout(real_name)

            return True

        return False

    for partition, skip_cudagraph in zip(
        reversed(partitions), reversed(skip_cudagraphs)
    ):
        output_names: OrderedSet[str] = OrderedSet()

        for node in partition:
            output_names.update(node.outputs_by_name.keys())

        returned_output_names = output_names.intersection(unmet_output_names)

        # all reads/writes are partition inputs except those generated
        # within the partition and tensor constants
        read_writes = dependencies.ReadWrites.merge_list(
            [node.read_writes for node in partition]
        )

        # WeakDep is fake dependency on unused buffer. It should not appear
        # in partition_input_names for inputs that are actually read or written.
        partition_input_names = (
            OrderedSet(
                [
                    x.name
                    for x in read_writes.reads | read_writes.writes
                    if not is_none_layout(x.name)
                ]
            )
            - output_names
        )

        partition_input_names = OrderedSet(
            self.mutation_real_name.get(name, name) for name in partition_input_names
        )

        buffer_names_to_free: OrderedSet[str] = OrderedSet()
        for node in partition:
            buffer_names_to_free.update(node.last_usage)

        # buffer_names_to_free may contain buffers allocated in previous
        # graph partitions. These buffers should also be a partition
        # input.
        extra_input_names = [
            name
            for name in (buffer_names_to_free - output_names)
            if name in name_to_node
        ]
        partition_input_names.update(extra_input_names)

        input_nodes = {
            name: name_to_node[name]
            for name in partition_input_names
            if name in name_to_node
        }
        input_deallocation = {
            name: name in buffer_names_to_free
            for name in partition_input_names
            if name in name_to_node
        }

        # if an input tensor is not freed in the partition function, it should
        # also be returned as an output. This brings benefits to cudagraph
        # since the returned output tensor is a cudagraph managed tensor with
        # a static tensor address.
        extra_output_names = [
            name
            for name in partition_input_names
            if name in name_to_node and name not in buffer_names_to_free
        ]

        returned_output_names.update(extra_output_names)

        returned_output_names = OrderedSet(
            self.mutation_real_name.get(name, name) for name in returned_output_names
        )

        output_nodes = [
            name_to_node[name]
            for name in returned_output_names
            if not is_none_layout(name)
        ]

        constant_names = [
            name for name in partition_input_names if name in V.graph.constants
        ]

        symbol_inputs = self.get_graph_partition_symbol_inputs(partition, input_nodes)

        partition_signature = GraphPartitionSignature(
            symbol_inputs,
            input_nodes,
            output_nodes,
            input_deallocation,
            skip_cudagraph,
            constant_names,
        )

        signatures.append(partition_signature)

        unmet_output_names = partition_input_names.union(
            unmet_output_names - returned_output_names
        )

    return signatures[::-1]

# ========================================
# torch 2.9 Inductor Scheduler monkeypatch
# ========================================
# This change monkeypatches a function in Inductor to work around the following
# bug: https://github.com/vllm-project/vllm/issues/26678
#
# The bug occurs when `use_inductor_graph_partition` is turned on and there
# exists operators inside of `splitting_ops` that have an in-place mutation. In
# vllm, this specifically occurs on the operator
# vllm.unified_attention_with_output. In this case, inductor does not populate
# the inductor IR's `origin_node` field, causing an assertion error when trying
# to access the node's `origin_node` field.
#
# So, we will monkeypatch torch._inductor.scheduler.Scheduler.should_partition
# so that it does not access the inductor IR node's `origin_node` field and just
# returns True if a node is registered as having a custom partition function.
# This is ok for now since vllm's implementation of the custom partition
# functions just return True.
# ========================================

def should_partition_patched(self, node, should_log: bool = False) -> bool:
    # This is a patched version of
    # torch._inductor.scheduler.Scheduler.should_partition that modifies
    # the following piece of code so that we always return True:
    # https://github.com/pytorch/pytorch/blob/ecb53078faf86ca1b33277df33b82985675bb011/torch/_inductor/scheduler.py#L4712-L4724
    (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
    files should be generated (to avoid biasing any benchmarks and pessimizing
    fusion decisions).
    from vllm.utils.torch_utils import is_torch_equal

    # Only apply the patch for torch 2.9.0 or 2.9.1
    if is_torch_equal("2.9.0") or is_torch_equal("2.9.1"):
        import builtins

        # Check if CUDA functionality is available without initializing CUDA
        # _cuda_getCurrentRawStream only exists in CUDA builds of PyTorch
        if hasattr(torch._C, "_cuda_getCurrentRawStream"):
            from torch._C import _cuda_getCurrentRawStream as _get_raw_stream

            builtins.get_raw_stream = _get_raw_stream  # type: ignore[attr-defined]

_patch_get_raw_stream_if_needed()

if is_torch_equal("2.9.0"):
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.graph import GraphLowering
    from torch.utils._config_module import _Config, _ConfigEntry

    # `custom_should_partition_ops` is a new config after 2.9.0. So this would
    # not overwrite any user configs.
    torch._inductor.config._config["custom_should_partition_ops"] = _ConfigEntry(
        _Config(default=[])
    )

    PythonWrapperCodegen.memory_plan_reuse = memory_plan_reuse_patched
    GraphLowering._update_scheduler = _update_scheduler_patched
