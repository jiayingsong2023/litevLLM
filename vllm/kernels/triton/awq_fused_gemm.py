# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from typing import Any

import torch

from vllm.triton_utils import tl, triton

_AWQ_FUSED_TUNING: dict[str, str] = {
    key: value
    for key, value in os.environ.items()
    if key.startswith("FASTINFERENCE_AWQ_")
}
_AWQ_FUSED_TUNING_LOCKED = False


def set_awq_fused_tuning_config(
    values: dict[str, object] | None, *, locked: bool = False
) -> None:
    """Install an immutable-ish FASTINFERENCE_AWQ_* config snapshot.

    LiteEngine calls this before model load so fused AWQ kernels do not read
    process environment variables on the inference hot path. Standalone tools
    retain import-time env snapshot behavior.
    """
    global _AWQ_FUSED_TUNING, _AWQ_FUSED_TUNING_LOCKED, _PERSISTENT_PROFILE_CACHE
    _AWQ_FUSED_TUNING = {
        str(key): str(value)
        for key, value in (values or {}).items()
        if str(key).startswith("FASTINFERENCE_AWQ_") and value is not None
    }
    _AWQ_FUSED_TUNING_LOCKED = bool(locked)
    _PERSISTENT_PROFILE_CACHE = None


def _env_get(name: str, default: str = "") -> str:
    if _AWQ_FUSED_TUNING_LOCKED:
        return _AWQ_FUSED_TUNING.get(name, default)
    return os.environ.get(name, _AWQ_FUSED_TUNING.get(name, default))


def _tool_override_get(name: str) -> str | None:
    if name in _AWQ_FUSED_TUNING:
        return _AWQ_FUSED_TUNING[name]
    if _AWQ_FUSED_TUNING_LOCKED:
        return None
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw


def _snapshot_override_get(name: str) -> str | None:
    raw = _AWQ_FUSED_TUNING.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw


def _tool_override_bool(
    raw: str | None,
    default: bool | None = None,
) -> bool | None:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _tool_override_int(
    raw: str | None,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _kernel_policy(
    config: Any | None = None, policy: dict[str, object] | None = None
) -> dict[str, object]:
    if isinstance(policy, dict):
        return policy
    if isinstance(config, dict):
        nested = config.get("kernel_policy")
        if isinstance(nested, dict):
            return nested
        return config
    kernel_policy = getattr(config, "kernel_policy", None)
    if isinstance(kernel_policy, dict):
        return kernel_policy
    return {}


def _kernel_policy_value(
    config: Any | None,
    policy: dict[str, object] | None,
    name: str,
    default: object = None,
) -> object:
    kernel_policy = _kernel_policy(config, policy)
    if name in kernel_policy:
        return kernel_policy[name]
    return default


def _policy_truthy(
    config: Any | None,
    policy: dict[str, object] | None,
    name: str,
    *,
    default: bool,
) -> bool:
    raw = _kernel_policy_value(config, policy, name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def _fused_gemm_dot_bf16_tool_override() -> str | None:
    return _tool_override_get("FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16")


def _resolve_use_bf16_dot(a: torch.Tensor, m: int, n: int) -> bool:
    """
    Whether to use bf16 operands in tl.dot (vs fp16).

    On ROCm, bf16 dot can be slower than fp16 dot for narrow decode
    (small M, moderate N); microbench: use fp16 dot there but still write
    bf16 output (USE_BF16_OUTPUT).

    Tool tuning override:
      - unset or "auto": heuristic (ROCm narrow decode -> fp16 dot;
        else bf16 when a is bf16)
      - "1"/"true"/"on": always bf16 dot when a is bf16
      - "0"/"false"/"off": always fp16 dot when a is bf16
    """
    if a.dtype != torch.bfloat16:
        return False
    raw = (_fused_gemm_dot_bf16_tool_override() or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # auto
    if getattr(torch.version, "hip", None) is not None:  # noqa: SIM102
        if m <= 4 and n <= 8192:
            return False
    return True


def _select_fused_gemm_blocks(
    m: int, n: int, _k: int
) -> tuple[int, int, int, int, int]:
    """
    Heuristic tile sizes for AWQ fused GEMM.

    Decode often uses M in {1..32}; fixed BLOCK_M=64 wastes most warps on the M axis.
    Prefill uses larger M where BLOCK_M=64 is appropriate.
    """
    k = int(_k)
    # Gemma4-31B MLP down_proj decode hot shape:
    #   n=5376, k=21504, m in {1,2}
    # This shape is latency-sensitive and generally prefers lower warp/stage pressure
    # compared with the broader decode defaults.
    mlp_down_decode_small_m = n == 5376 and k == 21504 and m <= 2
    block_k = 64
    deep_k_narrow_out = k >= 16384 and n <= 6144
    if m == 1:
        # M=1 decode: route to the GEMV specialization inside the kernel. A
        # BLOCK_M of 16 would waste 15/16 MFMA lanes on replicated rows; the
        # GEMV branch needs BLOCK_M==1 to be picked up (constexpr DCE of the
        # MMA path). Wider N tiles keep the reduction dominated by memory.
        block_m = 1
        if mlp_down_decode_small_m:
            block_n = 128
        elif n <= 8192 or k >= 16384:
            block_n = 256
        else:
            block_n = 128
    elif m <= 4:
        block_m = 16
        # Decode-heavy Gemma4 shapes prefer wider N tiles, but very wide output
        # projections (e.g. global q / gate-up) still favor 128.
        if mlp_down_decode_small_m:  # noqa: SIM114
            block_n = 128
        elif deep_k_narrow_out and m >= 4:
            block_n = 128
        elif n <= 8192 or (m == 1 and k >= 16384):
            block_n = 256
        else:
            block_n = 128
    elif m <= 32:
        block_m = 32
        if deep_k_narrow_out:  # noqa: SIM108
            block_n = 128
        else:
            block_n = 256 if n >= 4096 else 128
    elif m <= 192:
        # Gemma4 prefill-like batches consistently benefit from shorter M tiles
        # with a wider N tile on packed-int4 fused GEMM.
        if deep_k_narrow_out:
            block_m = 64
            block_n = 64
        else:
            block_m = 32
            block_n = 256 if n >= 4096 else 128
    else:
        block_m = 64
        # M=256 prefill-like shapes on Gemma4-31B favor square-ish N tiles.
        block_n = 64

    if mlp_down_decode_small_m:
        # m in {1,2} decode path: reduce occupancy pressure to improve single-request
        # token latency and stabilize C1 decode TPS.
        num_warps = 4
    elif block_n >= 128 or (m >= 64 and n >= 4096):
        num_warps = 8
    else:
        num_warps = 4
    num_stages = 1 if mlp_down_decode_small_m else 2
    return block_m, block_n, block_k, num_warps, num_stages


def _fused_gemm_blocks_tool_override(
    *, snapshot_only: bool = False
) -> tuple[int, int, int, int, int] | None:
    get_override = _snapshot_override_get if snapshot_only else _tool_override_get
    raw_m = get_override("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M")
    if raw_m is None:
        return None
    block_m = max(1, _tool_override_int(raw_m, 64, minimum=1, maximum=1 << 20))
    block_n = _tool_override_int(
        get_override("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N"),
        64,
        minimum=16,
        maximum=1 << 20,
    )
    block_k = _tool_override_int(
        get_override("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K"),
        32,
        minimum=8,
        maximum=1 << 20,
    )
    num_warps = _tool_override_int(
        get_override("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS"),
        4,
        minimum=1,
        maximum=8,
    )
    num_stages = _tool_override_int(
        get_override("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES"),
        2,
        minimum=1,
        maximum=4,
    )
    return block_m, block_n, block_k, num_warps, num_stages


_PERSISTENT_PROFILE_CACHE: dict[str, list[dict[str, int]]] | None = None


def _bundled_persistent_profile_path() -> Path:
    return Path(__file__).with_name("awq_fused_profile.json")


def _persistent_profile_path() -> Path:
    raw = _env_get("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", "").strip()
    if not raw:
        return _bundled_persistent_profile_path()
    if raw.lower() in ("0", "false", "no", "off"):
        return Path()
    return Path(raw).expanduser()


def _load_persistent_profile() -> dict[str, list[dict[str, int]]]:
    global _PERSISTENT_PROFILE_CACHE
    if _PERSISTENT_PROFILE_CACHE is not None:
        return _PERSISTENT_PROFILE_CACHE
    path = _persistent_profile_path()
    if not str(path) or (not path.is_file()):
        _PERSISTENT_PROFILE_CACHE = {}
        return _PERSISTENT_PROFILE_CACHE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _PERSISTENT_PROFILE_CACHE = {}
        return _PERSISTENT_PROFILE_CACHE
    out: dict[str, list[dict[str, int]]] = {}
    if isinstance(payload, dict):
        version = payload.get("version", 1)
        if version != 1:
            _PERSISTENT_PROFILE_CACHE = {}
            return _PERSISTENT_PROFILE_CACHE
        for key, value in payload.items():
            if key == "version":
                continue
            if isinstance(value, list):
                rows: list[dict[str, int]] = []
                for row in value:
                    if isinstance(row, dict):
                        rows.append(
                            {
                                str(k): int(v)
                                for k, v in row.items()
                                if isinstance(v, (int, float))
                            }
                        )
                out[str(key)] = rows
    _PERSISTENT_PROFILE_CACHE = out
    return _PERSISTENT_PROFILE_CACHE


def _match_profile_entry(
    row: dict[str, int],
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
) -> bool:
    m_min = int(row.get("m_min", 1))
    m_max = int(row.get("m_max", 1 << 30))
    if m < m_min or m > m_max:
        return False
    if "n" in row and int(row["n"]) != n:
        return False
    if "k" in row and int(row["k"]) != k:
        return False
    if "group_size" in row and int(row["group_size"]) != group_size:  # noqa: SIM103
        return False
    return True


def _lookup_persistent_blocks(
    kind: str,
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
) -> tuple[int, int, int, int, int] | None:
    rows = _load_persistent_profile().get(kind, [])
    for row in rows:
        if not _match_profile_entry(row, m=m, n=n, k=k, group_size=group_size):
            continue
        keys = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
        if all(key in row for key in keys):
            return (
                int(row["block_m"]),
                int(row["block_n"]),
                int(row["block_k"]),
                int(row["num_warps"]),
                int(row["num_stages"]),
            )
    return None


def _fused_gemm_autotune_tool_override() -> str | None:
    return _tool_override_get("FASTINFERENCE_AWQ_FUSED_AUTOTUNE")


def _fused_gemm_autotune_enabled(
    m: int,
    n: int,
    k: int,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    """
    Whether to launch Triton autotune for fused GEMM.

    Kernel policy / tool override:
      - "1"/"true"/"on": always autotune (except M==1; see below)
      - "0"/"false"/"off": never autotune
      - unset or "auto": shape-aware default

    The "auto" mode skips autotune on Gemma4-like production shapes where our
    curated heuristics are consistently faster and avoid warmup overhead.

    M==1 remains on the deterministic heuristic launcher. Isolated microbench
    results for BLOCK_M>=16 do not consistently transfer to whole-model Gemma4
    decode, where autotune choices have regressed latency.
    """
    mm, nn, kk = int(m), int(n), int(k)
    if mm == 1:
        return False
    raw = (
        str(_kernel_policy_value(config, policy, "awq_fused_autotune", "auto"))
        .strip()
        .lower()
    )
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # auto
    # Gemma4-31B decode dominant projections (m in {1,2,4}) benefit a lot from
    # autotuned tiles after the scale-index correctness fix; heuristic kernels
    # are kept for medium-M wide-output shapes to avoid unnecessary tuning cost.
    if mm <= 8 and nn >= 4096 and kk >= 4096:
        return True
    if mm <= 192 and nn >= 4096 and kk >= 4096:  # noqa: SIM103
        return False
    return True


def _env_fused_gemm_autotune_tool_override(
    m: int,
    n: int,
    k: int,
) -> bool:
    raw = (_fused_gemm_autotune_tool_override() or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return int(m) != 1
    return _fused_gemm_autotune_enabled(m, n, k)


def _fused_gemm_split_k_tool_override() -> str | None:
    return _tool_override_get("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K")


def _fused_gemm_split_k_snapshot_override() -> str | None:
    return _snapshot_override_get("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K")


def _select_split_k(
    m: int,
    n: int,
    k: int,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> int:
    """
    Split-K policy for fused int4 GEMM.

    Kernel policy / tool override:
      - integer >= 1: force value
      - unset/"auto": shape-aware default
    """
    raw_tool = _fused_gemm_split_k_tool_override()
    if raw_tool is not None:
        raw = raw_tool.strip().lower()
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    raw_policy = _kernel_policy_value(config, policy, "awq_fused_gemm_split_k", None)
    if raw_policy is not None:
        raw = str(raw_policy).strip().lower()
        if raw not in ("", "auto"):
            try:
                return max(1, int(raw))
            except ValueError:
                return 1

    raw_snapshot = _fused_gemm_split_k_snapshot_override()
    if raw_snapshot is not None:
        raw = raw_snapshot.strip().lower()
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    raw = (
        str(_kernel_policy_value(config, policy, "awq_fused_gemm_split_k", "auto"))
        .strip()
        .lower()
    )
    if raw not in ("", "auto"):
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    mm, nn, kk = int(m), int(n), int(k)
    if (
        awq_decode_gemv_enabled(config=config, policy=policy)
        and mm == 1
        and nn == 5376
        and kk == 21504
    ):
        # Gemma4-31B dense MLP down_proj decode hot shape. Route to split_k=1
        # so the M=1 grouped GEMV specialization can run; the generic split-K
        # kernel was previously chosen here to improve occupancy, but that kept
        # down_proj on the slower tiled path and is now superseded by the
        # decode-only GEMV fast path.
        return 1
    # Decode m=1 with deep-K benefits from K-splitting when there is not
    # enough work along N to saturate CU/SM occupancy.
    #
    # Shape families this heuristic covers (decode M=1):
    #   * Gemma4-31B / Qwen3.5-like gate_up fused pair: K~hidden (<=8k),
    #     N~2*intermediate (>=16k)   -> split_k=1 (lots of N-parallelism).
    #   * Gemma4-31B down_proj:     K~21504, N~hidden (~5k-6k) -> split_k=4.
    #     Narrow-N means few (pid_m, pid_n) tiles, so atomics over SPLIT_K
    #     are amortized by the extra CU fill.
    #   * Qwen3.5-9B style wide-N deep-K (K~16k, N>=8k)       -> split_k=4.
    #
    # The narrow-N lane uses a moderately higher K threshold (>= 20k) so
    # shallower-K decodes on Qwen3.5 (K~18944, N~3584) stay on split_k=1
    # where atomic overhead would otherwise dominate.
    if mm == 1 and kk >= 16384 and nn >= 8192:
        return 4
    if mm == 1 and kk >= 20480 and nn >= 4096:
        return 4
    return 1


def awq_decode_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_decode_gemv",
        default=False,
    )


def awq_fused_gate_up_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_fused_gate_up",
        default=False,
    )


def awq_o_proj_group32_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_o_proj_group32_gemv",
        default=True,
    )


def awq_group32_gemv_all_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_group32_gemv_all",
        default=False,
    )


def awq_qo_proj_exact_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_qo_proj_exact_gemv",
        default=False,
    )


def awq_q_proj_exact_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_q_proj_exact_gemv",
        default=False,
    )


def awq_o_proj_exact_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_o_proj_exact_gemv",
        default=False,
    )


def awq_o_proj_splitk_gemv_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    return _policy_truthy(
        config,
        policy,
        "awq_o_proj_splitk_gemv",
        default=False,
    )


def _o_proj_splitk_gemv_launch_config_tool_override() -> tuple[int, int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_SPLITK"),
            4,
            minimum=2,
            maximum=8,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_N"),
            128,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_SPLITK_BLOCK_GROUPS"),
            4,
            minimum=1,
            maximum=16,
        ),
    )


def _exact_gemv_launch_config_tool_override(n: int) -> tuple[int, int]:
    if n == 5376:
        return (
            _tool_override_int(
                _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_N"),
                256,
                minimum=32,
                maximum=512,
            ),
            _tool_override_int(
                _tool_override_get(
                    "FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_GROUPS"
                ),
                4,
                minimum=1,
                maximum=16,
            ),
        )
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_N"),
            128,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_Q_PROJ_GROUP32_GEMV_BLOCK_GROUPS"),
            8,
            minimum=1,
            maximum=16,
        ),
    )


def _o_proj_group32_gemv_launch_config_tool_override() -> tuple[int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_N"),
            256,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_O_PROJ_GROUP32_GEMV_BLOCK_GROUPS"),
            4,
            minimum=1,
            maximum=16,
        ),
    )


def _group32_gemv_all_launch_config_tool_override() -> tuple[int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_N"),
            128,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_GROUPS"),
            8,
            minimum=1,
            maximum=16,
        ),
    )


def _decode_gemv_launch_config_tool_override() -> tuple[int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_N"),
            128,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_PACKS"),
            16,
            minimum=4,
            maximum=64,
        ),
    )


def _fused_gate_up_launch_config_tool_override() -> tuple[int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_N"),
            128,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_FUSED_GATE_UP_BLOCK_PACKS"),
            16,
            minimum=4,
            maximum=64,
        ),
    )


def _qkv_group32_gemv_launch_config_tool_override() -> tuple[int, int]:
    return (
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_N"),
            256,
            minimum=32,
            maximum=512,
        ),
        _tool_override_int(
            _tool_override_get("FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_GROUPS"),
            4,
            minimum=1,
            maximum=16,
        ),
    )


def _resolve_fused_gemm_blocks(
    m: int,
    n: int,
    k: int,
) -> tuple[int, int, int, int, int]:
    override = _fused_gemm_blocks_tool_override(snapshot_only=True)
    if override is not None:
        return override
    return _select_fused_gemm_blocks(m, n, k)


def _resolve_packed_int4_fused_gemm_blocks(
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
    split_k: int,
) -> tuple[int, int, int, int, int]:
    override = _fused_gemm_blocks_tool_override(snapshot_only=True)
    if override is not None:
        return override
    profile_blocks = _lookup_persistent_blocks(
        "packed_int4_symmetric",
        m=m,
        n=n,
        k=k,
        group_size=group_size,
    )
    if profile_blocks is not None and split_k == 1 and m > 1:
        return profile_blocks
    return _select_fused_gemm_blocks(m, n, k)


# Packed int4 + AWQ native: tuned configs for ROCm (gfx1151-class) and CUDA;
# autotune picks best per key.
_PACKED_FUSED_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
    ),
    # Gemma4-31B packed-int4 winners on ROCm/gfx11-class workloads.
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2
    ),
]


@triton.jit
def _packed_int4_symmetric_grouped_gemv_m1(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    N,
    K,
    group_size,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cn,
    BLOCK_PACKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    M=1 symmetric packed-int4 GEMV for decode.

    Layout:
      a:       [1, K]
      b/qw:    [N, K / 8], one packed element stores 8 int4 values
      scales:  [N, K / group_size]

    The Gemma4 AWQ hot path uses group_size=32, so each quant group is exactly
    4 packed elements. This kernel keeps a 2D [BLOCK_N, BLOCK_PACKS] tile to
    expose enough parallelism while loading each packed element once instead
    of 8 times.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    num_packs = tl.cdiv(K, 8)
    offs_p = tl.arange(0, BLOCK_PACKS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < num_packs
        packed = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + pack_idx[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        group_idx = pack_idx // 4
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.zeros((BLOCK_N, BLOCK_PACKS), dtype=tl.float32)
        for nibble in tl.static_range(0, 8):
            k_idx = pack_idx * 8 + nibble
            aval = tl.load(
                a_ptr + k_idx * stride_ak, mask=mask_p & (k_idx < K), other=0.0
            )
            q = (packed >> (nibble * 4)) & 0xF
            partial += aval[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
        acc += tl.sum(partial, axis=1)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals.to(tl.float32)

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)

    c_ptrs = c_ptr + offs_n * stride_cn
    tl.store(c_ptrs, out, mask=mask_n)


@triton.jit
def _packed_int4_symmetric_group32_gemv_m1(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    N,
    K,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cn,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    M=1 symmetric packed-int4 GEMV specialized for group_size=32.

    Each quant group has one scale and exactly four uint32 packed elements
    (4 packs * 8 nibbles = 32 K values). The generic GEMV loops over packs
    and reloads the same scale four times. This kernel loops over quant groups
    and reuses the scale while consuming the four packs in that group.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    num_groups = tl.cdiv(K, 32)
    offs_g = tl.arange(0, BLOCK_GROUPS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group_base in range(0, tl.cdiv(num_groups, BLOCK_GROUPS)):
        group_idx = group_base * BLOCK_GROUPS + offs_g
        mask_g = group_idx < num_groups
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sk,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        group_partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), dtype=tl.float32)
        for pack_in_group in tl.static_range(0, 4):
            pack_idx = group_idx * 4 + pack_in_group
            packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + pack_idx[None, :] * stride_bk,
                mask=mask_n[:, None] & mask_g[None, :],
                other=0,
            ).to(tl.int32)
            for nibble in tl.static_range(0, 8):
                k_idx = group_idx * 32 + pack_in_group * 8 + nibble
                aval = tl.load(
                    a_ptr + k_idx * stride_ak, mask=mask_g & (k_idx < K), other=0.0
                )
                q = (packed >> (nibble * 4)) & 0xF
                group_partial += (
                    aval[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
        acc += tl.sum(group_partial, axis=1)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals.to(tl.float32)

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)

    tl.store(c_ptr + offs_n * stride_cn, out, mask=mask_n)


@triton.jit
def _packed_int4_symmetric_group32_gemv_m1_exact(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cn,
    K_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Exact-shape M=1 group32 GEMV for Gemma4 q_proj/o_proj decode.

    This variant is only dispatched for known 31B attention projection shapes
    where N is divisible by BLOCK_N and K is divisible by 32. It removes the
    hot loop's N/K boundary masks and keeps the runtime kernel signature small.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_g = tl.arange(0, BLOCK_GROUPS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group_base in range(0, K_GROUPS // BLOCK_GROUPS):
        group_idx = group_base * BLOCK_GROUPS + offs_g
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sk,
        ).to(tl.float32)
        group_partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), dtype=tl.float32)
        for pack_in_group in tl.static_range(0, 4):
            pack_idx = group_idx * 4 + pack_in_group
            packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + pack_idx[None, :] * stride_bk,
            ).to(tl.int32)
            for nibble in tl.static_range(0, 8):
                k_idx = group_idx * 32 + pack_in_group * 8 + nibble
                aval = tl.load(a_ptr + k_idx * stride_ak)
                q = (packed >> (nibble * 4)) & 0xF
                group_partial += (
                    aval[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
        acc += tl.sum(group_partial, axis=1)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n)
        acc += bias_vals.to(tl.float32)

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)

    tl.store(c_ptr + offs_n * stride_cn, out)


@triton.jit
def _packed_int4_symmetric_group32_o_proj_splitk_partial_m1(
    a_ptr,
    b_ptr,
    s_ptr,
    partial_ptr,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    N: tl.constexpr,
    K_GROUPS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Gemma4-31B o_proj M=1 split-K partial GEMV.

    Grid is [N tile, K split]. Each program accumulates a contiguous K-group
    slice and writes fp32 partial[SPLIT_K, N]. A second kernel reduces partials.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    groups_per_split = K_GROUPS // SPLIT_K
    group_start = pid_k * groups_per_split
    offs_g = tl.arange(0, BLOCK_GROUPS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group_iter in range(0, groups_per_split // BLOCK_GROUPS):
        group_idx = group_start + group_iter * BLOCK_GROUPS + offs_g
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sk,
        ).to(tl.float32)
        group_partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), dtype=tl.float32)
        for pack_in_group in tl.static_range(0, 4):
            pack_idx = group_idx * 4 + pack_in_group
            packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + pack_idx[None, :] * stride_bk,
            ).to(tl.int32)
            for nibble in tl.static_range(0, 8):
                k_idx = group_idx * 32 + pack_in_group * 8 + nibble
                aval = tl.load(a_ptr + k_idx * stride_ak)
                q = (packed >> (nibble * 4)) & 0xF
                group_partial += (
                    aval[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
        acc += tl.sum(group_partial, axis=1)

    tl.store(partial_ptr + pid_k * N + offs_n, acc)


@triton.jit
def _packed_int4_symmetric_group32_o_proj_splitk_reduce_m1(
    partial_ptr,
    c_ptr,
    bias_ptr,
    stride_cn,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for split_idx in tl.static_range(0, SPLIT_K):
        acc += tl.load(partial_ptr + split_idx * N + offs_n).to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)
    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(c_ptr + offs_n * stride_cn, out)


@triton.jit
def _packed_int4_symmetric_group32_qkv_m1(
    a_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    qs_ptr,
    ks_ptr,
    vs_ptr,
    c_ptr,
    QN: tl.constexpr,
    KN: tl.constexpr,
    VN: tl.constexpr,
    K: tl.constexpr,
    stride_ak,
    stride_qn,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_qsn,
    stride_qsk,
    stride_ksn,
    stride_ksk,
    stride_vsn,
    stride_vsk,
    stride_cn,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_V: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    M=1 fused q/k/v packed-int4 GEMV for Gemma4 decode.

    Q, K and optional V share the same activation row and group_size=32. This
    stores [q | k | v] into one contiguous output buffer, reducing three GEMV
    launches and repeated activation reads to one launch.
    """
    total_n = QN + KN + VN
    pid_n = tl.program_id(0)
    offs_t = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_t = offs_t < total_n

    is_q = offs_t < QN
    is_k = (offs_t >= QN) & (offs_t < (QN + KN))
    local_n = tl.where(is_q, offs_t, tl.where(is_k, offs_t - QN, offs_t - QN - KN))

    num_groups = tl.cdiv(K, 32)
    offs_g = tl.arange(0, BLOCK_GROUPS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group_base in range(0, tl.cdiv(num_groups, BLOCK_GROUPS)):
        group_idx = group_base * BLOCK_GROUPS + offs_g
        mask_g = group_idx < num_groups
        q_scale = tl.load(
            qs_ptr + local_n[:, None] * stride_qsn + group_idx[None, :] * stride_qsk,
            mask=mask_t[:, None] & is_q[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        k_scale = tl.load(
            ks_ptr + local_n[:, None] * stride_ksn + group_idx[None, :] * stride_ksk,
            mask=mask_t[:, None] & is_k[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            vs_ptr + local_n[:, None] * stride_vsn + group_idx[None, :] * stride_vsk,
            mask=mask_t[:, None]
            & (~is_q[:, None])
            & (~is_k[:, None])
            & HAS_V
            & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        scale = q_scale + k_scale + v_scale
        group_partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), dtype=tl.float32)
        for pack_in_group in tl.static_range(0, 4):
            pack_idx = group_idx * 4 + pack_in_group
            q_packed = tl.load(
                q_ptr + local_n[:, None] * stride_qn + pack_idx[None, :] * stride_qk,
                mask=mask_t[:, None] & is_q[:, None] & mask_g[None, :],
                other=0,
            ).to(tl.int32)
            k_packed = tl.load(
                k_ptr + local_n[:, None] * stride_kn + pack_idx[None, :] * stride_kk,
                mask=mask_t[:, None] & is_k[:, None] & mask_g[None, :],
                other=0,
            ).to(tl.int32)
            v_packed = tl.load(
                v_ptr + local_n[:, None] * stride_vn + pack_idx[None, :] * stride_vk,
                mask=mask_t[:, None]
                & (~is_q[:, None])
                & (~is_k[:, None])
                & HAS_V
                & mask_g[None, :],
                other=0,
            ).to(tl.int32)
            packed = q_packed + k_packed + v_packed
            for nibble in tl.static_range(0, 8):
                k_idx = group_idx * 32 + pack_in_group * 8 + nibble
                aval = tl.load(
                    a_ptr + k_idx * stride_ak, mask=mask_g & (k_idx < K), other=0.0
                )
                q = (packed >> (nibble * 4)) & 0xF
                group_partial += (
                    aval[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
        acc += tl.sum(group_partial, axis=1)

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(c_ptr + offs_t * stride_cn, out, mask=mask_t)


@triton.jit
def _packed_int4_symmetric_fused_gate_up_m1(
    a_ptr,
    gate_ptr,
    up_ptr,
    gate_s_ptr,
    up_s_ptr,
    c_ptr,
    N,
    K,
    group_size,
    stride_ak,
    stride_gn,
    stride_gk,
    stride_un,
    stride_uk,
    stride_gsn,
    stride_gsk,
    stride_usn,
    stride_usk,
    stride_cn,
    BLOCK_PACKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACT_KIND: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    M=1 fused gate/up decode for symmetric packed-int4 MLPs.

    Computes:
      out[n] = activation(x @ W_gate[n]) * (x @ W_up[n])

    The activation vector is shared by both GEMVs, so this avoids one kernel
    launch and the 2I intermediate tensor produced by concat gate/up fusion.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    num_packs = tl.cdiv(K, 8)
    offs_p = tl.arange(0, BLOCK_PACKS)
    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < num_packs
        group_idx = pack_idx // 4
        gate_packed = tl.load(
            gate_ptr + offs_n[:, None] * stride_gn + pack_idx[None, :] * stride_gk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        up_packed = tl.load(
            up_ptr + offs_n[:, None] * stride_un + pack_idx[None, :] * stride_uk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        gate_scale = tl.load(
            gate_s_ptr + offs_n[:, None] * stride_gsn + group_idx[None, :] * stride_gsk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            up_s_ptr + offs_n[:, None] * stride_usn + group_idx[None, :] * stride_usk,
            mask=mask_n[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_partial = tl.zeros((BLOCK_N, BLOCK_PACKS), dtype=tl.float32)
        up_partial = tl.zeros((BLOCK_N, BLOCK_PACKS), dtype=tl.float32)
        for nibble in tl.static_range(0, 8):
            k_idx = pack_idx * 8 + nibble
            aval = tl.load(
                a_ptr + k_idx * stride_ak, mask=mask_p & (k_idx < K), other=0.0
            )
            gate_q = (gate_packed >> (nibble * 4)) & 0xF
            up_q = (up_packed >> (nibble * 4)) & 0xF
            gate_partial += (
                aval[None, :].to(tl.float32)
                * (gate_q.to(tl.float32) - 8.0)
                * gate_scale
            )
            up_partial += (
                aval[None, :].to(tl.float32) * (up_q.to(tl.float32) - 8.0) * up_scale
            )
        gate_acc += tl.sum(gate_partial, axis=1)
        up_acc += tl.sum(up_partial, axis=1)

    if ACT_KIND == 1:
        # GELU tanh approximation used by Gemma-style gelu_pytorch_tanh.
        # tanh(x) is not available in this Triton build; use
        # 0.5 * (1 + tanh(x)) == sigmoid(2x).
        x3 = gate_acc * gate_acc * gate_acc
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
        act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
    else:
        act = gate_acc / (1.0 + tl.exp(-gate_acc))
    out = act * up_acc

    out_cast = out.to(tl.bfloat16) if USE_BF16_OUTPUT else out.to(tl.float16)
    tl.store(c_ptr + offs_n * stride_cn, out_cast, mask=mask_n)


@triton.jit
def _awq_native_tiled_gemm_split_k(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Split-K AWQ GEMM Kernel for small M.
    Parallelizes reduction over K dimension to increase GPU utilization.
    """
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # Split K dimension
    iters_per_sk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    start_k = pid_sk * iters_per_sk * BLOCK_K
    end_k = min(K, (pid_sk + 1) * iters_per_sk * BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_bn[:, None] * stride_bn + ((start_k + offs_k[None, :]) // 8) * stride_bk
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, iters_per_sk):
        curr_k_base = start_k + k * BLOCK_K
        mask_k = (offs_k[None, :] < BLOCK_K) & (curr_k_base + offs_k[None, :] < K)
        mask_k = mask_k & (curr_k_base < end_k)

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = curr_k_base + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (
            offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk
        )

        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )
        z_packed = tl.load(
            z_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=0,
        )
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        # Same M=1 GEMV rationale as ``_packed_int4_symmetric_tiled_gemm_split_k``:
        # MFMA tiles with BLOCK_M>1 waste lanes when M==1 at runtime.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
            if USE_BF16_DOT:
                accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
            else:
                accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    result = (
        accumulator.to(tl.bfloat16) if USE_BF16_OUTPUT else accumulator.to(tl.float16)
    )

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Use atomic add if SPLIT_K > 1
    if SPLIT_K > 1:
        tl.atomic_add(
            c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        )
    else:
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
            accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
            result = (
                accumulator.to(tl.bfloat16)
                if USE_BF16_OUTPUT
                else accumulator.to(tl.float16)
            )
        tl.store(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _awq_native_tiled_gemm_heuristic(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Native Triton Tiling Implementation for AMD gfx1151.
    Performs K-axis splitting inside a single kernel launch to avoid TLB storm.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # Static K-axis offsets
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Dimensionality Reduction: The main loop IS the tiling logic
    # By controlling BLOCK_K and the number of iterations, we keep MC stable
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        # 1. Load A
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        # 2. Load and Dequantize B
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        # 3. Load Scales/Zeros
        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (
            offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk
        )

        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )
        z_packed = tl.load(
            z_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=0,
        )
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        # 4. Multiply-Accumulate: M=1 GEMV path matches packed-int4 kernels.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
            if USE_BF16_DOT:
                accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
            else:
                accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    # Store Result (output dtype may be bf16 even when tl.dot used fp16 operands)
    c = accumulator.to(tl.bfloat16) if USE_BF16_OUTPUT else accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _packed_int4_symmetric_tiled_gemm_split_k(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    iters_per_sk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    start_k = pid_sk * iters_per_sk * BLOCK_K
    end_k = min(K, (pid_sk + 1) * iters_per_sk * BLOCK_K)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_bn[:, None] * stride_bn + ((start_k + offs_k[None, :]) // 8) * stride_bk
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, iters_per_sk):
        curr_k_base = start_k + k * BLOCK_K
        mask_k = (offs_k[None, :] < BLOCK_K) & (curr_k_base + offs_k[None, :] < K)
        mask_k = mask_k & (curr_k_base < end_k)

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = curr_k_base + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        # ---- M=1 GEMV specialization -----------------------------------
        # Decode batches with M=1 produce a [1, BLOCK_K] activation tile. The
        # generic MMA path forces BLOCK_M>=16 (MFMA lane requirement), which
        # wastes 15/16 SIMD lanes on replicated inputs. When BLOCK_M==1 we
        # broadcast multiply-add along the N axis in fp32 to keep every lane
        # productive without sacrificing accuracy.
        #
        #   a : [1,        BLOCK_K]  (activation row)
        #   b : [BLOCK_N,  BLOCK_K]  (dequantized weights, fp32)
        # a * b broadcasts to [BLOCK_N, BLOCK_K]; tl.sum over K yields [BLOCK_N].
        # ----------------------------------------------------------------
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
            if USE_BF16_DOT:
                accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
            else:
                accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    result = (
        accumulator.to(tl.bfloat16) if USE_BF16_OUTPUT else accumulator.to(tl.float16)
    )

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    if SPLIT_K > 1:
        tl.atomic_add(
            c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        )
    else:
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
            accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
            result = (
                accumulator.to(tl.bfloat16)
                if USE_BF16_OUTPUT
                else accumulator.to(tl.float16)
            )
        tl.store(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _packed_int4_symmetric_tiled_gemm_heuristic(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Optimized Symmetric INT4 GEMM.
    Uses per-element group index for scales so results remain correct when
    BLOCK_K spans multiple quant groups (e.g. group_size=32, BLOCK_K=64).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Heuristic: if BLOCK_K is a multiple of group_size (or vice versa),
    # we can optimize scale loads. Most common: BLOCK_K=64, group_size=128.
    # Scale stays same for 2 iterations.

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)

        # Fast 4-bit unpacking using bitwise shifts
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        # (q - 8) * scale
        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)

        # See split-k kernel above for the M=1 GEMV rationale: broadcast-FMA
        # along N with fp32 accumulation keeps SIMD lanes fully utilized when
        # MFMA's 16-row tile would otherwise waste 15/16 lanes.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
            if USE_BF16_DOT:
                accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
            else:
                accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]

    c = accumulator.to(tl.bfloat16) if USE_BF16_OUTPUT else accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    configs=_PACKED_FUSED_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "BF16_DOT", "OUT_BF16", "HAS_BIAS"],
    warmup=8,
    rep=20,
    cache_results=True,
)
@triton.jit
def _packed_int4_symmetric_tiled_gemm_autotuned(
    a_ptr,
    b_ptr,
    s_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BF16_DOT: tl.constexpr,
    OUT_BF16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0x0F

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        # See split-k kernel for M=1 GEMV rationale.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
            if BF16_DOT:  # noqa: SIM108
                accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
            else:
                accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    c = accumulator.to(tl.bfloat16) if OUT_BF16 else accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    configs=_PACKED_FUSED_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "BF16_DOT", "OUT_BF16", "HAS_BIAS"],
    warmup=8,
    rep=20,
    cache_results=True,
)
@triton.jit
def _awq_native_tiled_gemm_autotuned(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BF16_DOT: tl.constexpr,
    OUT_BF16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0x0F

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (
            offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk
        )

        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )
        z_packed = tl.load(
            z_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=0,
        )
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0x0F

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        if BF16_DOT:  # noqa: SIM108
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    c = accumulator.to(tl.bfloat16) if OUT_BF16 else accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def awq_fused_gemm(
    a,
    qweight,
    scales,
    qzeros,
    group_size,
    out=None,
    bias=None,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
):
    M, K = a.shape
    N = qweight.shape[0]

    has_bias = bias is not None
    if has_bias:
        if bias.dim() != 1 or int(bias.shape[0]) != N:
            raise ValueError(
                f"awq_fused_gemm: bias must be 1D of size N={N}, "
                f"got {tuple(bias.shape)}"
            )
        bias = bias.contiguous().reshape(N)
        if bias.device != a.device:
            raise ValueError(
                "awq_fused_gemm: bias must be on the same device as activations"
            )
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"awq_fused_gemm: unsupported bias dtype {bias.dtype}")
    bias_ptr_arg = bias if has_bias else a

    use_bf16_dot = _resolve_use_bf16_dot(a, M, N)
    use_bf16_output = a.dtype == torch.bfloat16
    bf16_dot = 1 if use_bf16_dot else 0
    out_bf16 = 1 if use_bf16_output else 0

    # Decision for Split-K: Only for extremely narrow M and deep K
    split_k = _select_split_k(M, N, K, config=config, policy=policy)

    if out is None:
        if split_k > 1:
            # Atomic add requires zeros
            c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
        if split_k > 1:
            c.zero_()

    if (
        _fused_gemm_autotune_enabled(M, N, K, config=config, policy=policy)
        and split_k == 1
        and M > 1
    ):
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        try:
            _awq_native_tiled_gemm_autotuned[grid](
                a,
                qweight,
                scales,
                qzeros,
                c,
                bias_ptr_arg,
                M,
                N,
                K,
                group_size,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                qzeros.stride(0),
                qzeros.stride(1) if qzeros is not None else 0,
                c.stride(0),
                c.stride(1),
                BF16_DOT=bf16_dot,
                OUT_BF16=out_bf16,
                HAS_BIAS=has_bias,
            )
            return c
        except Exception:
            pass

    block_m, block_n, block_k, num_warps, num_stages = _resolve_fused_gemm_blocks(
        M, N, K
    )

    if split_k > 1:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k)
        _awq_native_tiled_gemm_split_k[grid](
            a,
            qweight,
            scales,
            qzeros,
            c,
            bias_ptr_arg,
            M,
            N,
            K,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            qzeros.stride(0),
            qzeros.stride(1) if qzeros is not None else 0,
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        _awq_native_tiled_gemm_heuristic[grid](
            a,
            qweight,
            scales,
            qzeros,
            c,
            bias_ptr_arg,
            M,
            N,
            K,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            qzeros.stride(0),
            qzeros.stride(1) if qzeros is not None else 0,
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return c


def packed_int4_symmetric_fused_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> torch.Tensor:
    m, k = a.shape
    n = qweight.shape[0]

    has_bias = bias is not None
    if has_bias:
        if bias.dim() != 1 or int(bias.shape[0]) != n:
            raise ValueError(
                "packed_int4_symmetric_fused_gemm: bias must be 1D of "
                f"size N={n}, got {tuple(bias.shape)}"
            )
        bias = bias.contiguous().reshape(n)
        if bias.device != a.device:
            raise ValueError(
                "packed_int4_symmetric_fused_gemm: bias must be on the "
                "same device as activations"
            )
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"packed_int4_symmetric_fused_gemm: unsupported bias dtype {bias.dtype}"
            )
    bias_ptr_arg = bias if has_bias else a

    use_bf16_dot = _resolve_use_bf16_dot(a, m, n)
    use_bf16_output = a.dtype == torch.bfloat16
    bf16_dot = 1 if use_bf16_dot else 0
    out_bf16 = 1 if use_bf16_output else 0

    split_k = _select_split_k(m, n, k, config=config, policy=policy)

    if out is None:
        if split_k > 1:
            c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    else:
        c = out
        if split_k > 1:
            c.zero_()

    if (
        awq_decode_gemv_enabled(config=config, policy=policy)
        and m == 1
        and split_k == 1
        and int(group_size) == 32
        and k % 32 == 0
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and c.is_contiguous()
    ):
        if (
            awq_o_proj_splitk_gemv_enabled(config=config, policy=policy)
            and n == 5376
            and k == 16384
            and (k // 32) % 2 == 0
        ):
            o_split_k, block_n, block_groups = (
                _o_proj_splitk_gemv_launch_config_tool_override()
            )
            k_groups = k // 32
            if (
                n % block_n == 0
                and k_groups % o_split_k == 0
                and (k_groups // o_split_k) % block_groups == 0
            ):
                partial = torch.empty(
                    (o_split_k, n), device=a.device, dtype=torch.float32
                )
                partial_grid = (triton.cdiv(n, block_n), o_split_k)
                _packed_int4_symmetric_group32_o_proj_splitk_partial_m1[partial_grid](
                    a,
                    qweight,
                    scales,
                    partial,
                    a.stride(1),
                    qweight.stride(0),
                    qweight.stride(1),
                    scales.stride(0),
                    scales.stride(1),
                    N=n,
                    K_GROUPS=k_groups,
                    SPLIT_K=o_split_k,
                    BLOCK_GROUPS=block_groups,
                    BLOCK_N=block_n,
                    num_warps=8 if block_n >= 128 else 4,
                    num_stages=1,
                )
                reduce_grid = (triton.cdiv(n, block_n),)
                _packed_int4_symmetric_group32_o_proj_splitk_reduce_m1[reduce_grid](
                    partial,
                    c,
                    bias_ptr_arg,
                    c.stride(1),
                    N=n,
                    SPLIT_K=o_split_k,
                    BLOCK_N=block_n,
                    USE_BF16_OUTPUT=use_bf16_output,
                    HAS_BIAS=has_bias,
                    num_warps=8 if block_n >= 128 else 4,
                    num_stages=1,
                )
                return c

        use_q_exact = (
            n == 16384
            and k == 5376
            and (
                awq_qo_proj_exact_gemv_enabled(config=config, policy=policy)
                or awq_q_proj_exact_gemv_enabled(config=config, policy=policy)
            )
        )
        use_o_exact = (
            n == 5376
            and k == 16384
            and (
                awq_qo_proj_exact_gemv_enabled(config=config, policy=policy)
                or awq_o_proj_exact_gemv_enabled(config=config, policy=policy)
            )
        )
        if use_q_exact or use_o_exact:
            block_n, block_groups = _exact_gemv_launch_config_tool_override(n)
            if n % block_n == 0 and (k // 32) % block_groups == 0:
                grid = (triton.cdiv(n, block_n),)
                _packed_int4_symmetric_group32_gemv_m1_exact[grid](
                    a,
                    qweight,
                    scales,
                    c,
                    bias_ptr_arg,
                    a.stride(1),
                    qweight.stride(0),
                    qweight.stride(1),
                    scales.stride(0),
                    scales.stride(1),
                    c.stride(1),
                    K_GROUPS=k // 32,
                    BLOCK_GROUPS=block_groups,
                    BLOCK_N=block_n,
                    USE_BF16_OUTPUT=use_bf16_output,
                    HAS_BIAS=has_bias,
                    num_warps=8 if block_n >= 128 else 4,
                    num_stages=1,
                )
                return c

        if (
            awq_o_proj_group32_gemv_enabled(config=config, policy=policy)
            and n == 5376
            and k == 16384
        ):
            block_n, block_groups = _o_proj_group32_gemv_launch_config_tool_override()
            grid = (triton.cdiv(n, block_n),)
            _packed_int4_symmetric_group32_gemv_m1[grid](
                a,
                qweight,
                scales,
                c,
                bias_ptr_arg,
                n,
                k,
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                c.stride(1),
                BLOCK_GROUPS=block_groups,
                BLOCK_N=block_n,
                USE_BF16_OUTPUT=use_bf16_output,
                HAS_BIAS=has_bias,
                num_warps=8 if block_n >= 128 else 4,
                num_stages=1,
            )
            return c

        if awq_group32_gemv_all_enabled(config=config, policy=policy):
            block_n, block_groups = _group32_gemv_all_launch_config_tool_override()
            grid = (triton.cdiv(n, block_n),)
            _packed_int4_symmetric_group32_gemv_m1[grid](
                a,
                qweight,
                scales,
                c,
                bias_ptr_arg,
                n,
                k,
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                c.stride(1),
                BLOCK_GROUPS=block_groups,
                BLOCK_N=block_n,
                USE_BF16_OUTPUT=use_bf16_output,
                HAS_BIAS=has_bias,
                num_warps=8 if block_n >= 128 else 4,
                num_stages=1,
            )
            return c

        block_n, block_packs = _decode_gemv_launch_config_tool_override()
        grid = (triton.cdiv(n, block_n),)
        _packed_int4_symmetric_grouped_gemv_m1[grid](
            a,
            qweight,
            scales,
            c,
            bias_ptr_arg,
            n,
            k,
            group_size,
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            c.stride(1),
            BLOCK_PACKS=block_packs,
            BLOCK_N=block_n,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=8 if block_n >= 128 else 4,
            num_stages=1,
        )
        return c

    if (
        _fused_gemm_autotune_enabled(m, n, k, config=config, policy=policy)
        and split_k == 1
        and m > 1
    ):
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )
        try:
            _packed_int4_symmetric_tiled_gemm_autotuned[grid](
                a,
                qweight,
                scales,
                c,
                bias_ptr_arg,
                m,
                n,
                k,
                group_size,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                c.stride(0),
                c.stride(1),
                BF16_DOT=bf16_dot,
                OUT_BF16=out_bf16,
                HAS_BIAS=has_bias,
            )
            return c
        except Exception:
            pass

    block_m, block_n, block_k, num_warps, num_stages = (
        _resolve_packed_int4_fused_gemm_blocks(
            m=m,
            n=n,
            k=k,
            group_size=int(group_size),
            split_k=split_k,
        )
    )

    if split_k > 1:
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), split_k)
        _packed_int4_symmetric_tiled_gemm_split_k[grid](
            a,
            qweight,
            scales,
            c,
            bias_ptr_arg,
            m,
            n,
            k,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _packed_int4_symmetric_tiled_gemm_heuristic[grid](
            a,
            qweight,
            scales,
            c,
            bias_ptr_arg,
            m,
            n,
            k,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return c


def packed_int4_symmetric_fused_gate_up_m1(
    a: torch.Tensor,
    gate_qweight: torch.Tensor,
    up_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    up_scales: torch.Tensor,
    group_size: int,
    *,
    activation: str = "silu",
    out: torch.Tensor | None = None,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> torch.Tensor:
    m, k = a.shape
    n = gate_qweight.shape[0]
    if m != 1:
        raise ValueError("packed_int4_symmetric_fused_gate_up_m1: requires M == 1")
    if int(group_size) != 32 or k % 32 != 0:
        raise ValueError(
            "packed_int4_symmetric_fused_gate_up_m1: requires group_size=32 and K%32==0"
        )
    if tuple(gate_qweight.shape) != tuple(up_qweight.shape):
        raise ValueError(
            "packed_int4_symmetric_fused_gate_up_m1: qweight shape mismatch"
        )
    if tuple(gate_scales.shape) != tuple(up_scales.shape):
        raise ValueError("packed_int4_symmetric_fused_gate_up_m1: scale shape mismatch")
    if int(gate_qweight.shape[1]) * 8 != k:
        raise ValueError("packed_int4_symmetric_fused_gate_up_m1: K mismatch")
    c = torch.empty((1, n), device=a.device, dtype=a.dtype) if out is None else out
    del config, policy
    block_n, block_packs = _fused_gate_up_launch_config_tool_override()
    act = str(activation).lower()
    act_kind = 1 if act in ("gelu", "gelu_pytorch_tanh") else 0
    grid = (triton.cdiv(n, block_n),)
    _packed_int4_symmetric_fused_gate_up_m1[grid](
        a,
        gate_qweight,
        up_qweight,
        gate_scales,
        up_scales,
        c,
        n,
        k,
        group_size,
        a.stride(1),
        gate_qweight.stride(0),
        gate_qweight.stride(1),
        up_qweight.stride(0),
        up_qweight.stride(1),
        gate_scales.stride(0),
        gate_scales.stride(1),
        up_scales.stride(0),
        up_scales.stride(1),
        c.stride(1),
        BLOCK_PACKS=block_packs,
        BLOCK_N=block_n,
        ACT_KIND=act_kind,
        USE_BF16_OUTPUT=a.dtype == torch.bfloat16,
        num_warps=8 if block_n >= 128 else 4,
        num_stages=1,
    )
    return c


def packed_int4_symmetric_fused_qkv_m1(
    a: torch.Tensor,
    q_qweight: torch.Tensor,
    k_qweight: torch.Tensor,
    v_qweight: torch.Tensor | None,
    q_scales: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor | None,
    group_size: int,
    *,
    out: torch.Tensor | None = None,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> torch.Tensor:
    m, k = a.shape
    if m != 1:
        raise ValueError("packed_int4_symmetric_fused_qkv_m1: requires M == 1")
    if int(group_size) != 32 or k % 32 != 0:
        raise ValueError(
            "packed_int4_symmetric_fused_qkv_m1: requires group_size=32 and K%32==0"
        )
    qn = int(q_qweight.shape[0])
    kn = int(k_qweight.shape[0])
    has_v = v_qweight is not None and v_scales is not None
    vn = int(v_qweight.shape[0]) if has_v else 0
    total_n = qn + kn + vn
    if out is None:
        c = torch.empty((1, total_n), device=a.device, dtype=a.dtype)
    else:
        c = out
    if v_qweight is None:
        v_qweight = q_qweight
    if v_scales is None:
        v_scales = q_scales
    del config, policy
    block_n, block_groups = _qkv_group32_gemv_launch_config_tool_override()
    grid = (triton.cdiv(total_n, block_n),)
    _packed_int4_symmetric_group32_qkv_m1[grid](
        a,
        q_qweight,
        k_qweight,
        v_qweight,
        q_scales,
        k_scales,
        v_scales,
        c,
        QN=qn,
        KN=kn,
        VN=vn,
        K=k,
        stride_ak=a.stride(1),
        stride_qn=q_qweight.stride(0),
        stride_qk=q_qweight.stride(1),
        stride_kn=k_qweight.stride(0),
        stride_kk=k_qweight.stride(1),
        stride_vn=v_qweight.stride(0),
        stride_vk=v_qweight.stride(1),
        stride_qsn=q_scales.stride(0),
        stride_qsk=q_scales.stride(1),
        stride_ksn=k_scales.stride(0),
        stride_ksk=k_scales.stride(1),
        stride_vsn=v_scales.stride(0),
        stride_vsk=v_scales.stride(1),
        stride_cn=c.stride(1),
        BLOCK_GROUPS=block_groups,
        BLOCK_N=block_n,
        HAS_V=has_v,
        USE_BF16_OUTPUT=a.dtype == torch.bfloat16,
        num_warps=8 if block_n >= 128 else 4,
        num_stages=1,
    )
    return c


def packed_int4_symmetric_fused_qkv_m1_safe(
    a: torch.Tensor,
    q_qweight: torch.Tensor,
    k_qweight: torch.Tensor,
    v_qweight: torch.Tensor | None,
    q_scales: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor | None,
    group_size: int,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if not awq_decode_gemv_enabled(config=config, policy=policy):
        return a, False, "decode_gemv_disabled"
    if a.dim() != 2 or int(a.shape[0]) != 1:
        return a, False, "input_not_m1_2d"
    if q_qweight.dim() != 2 or k_qweight.dim() != 2:
        return a, False, "qk_weight_not_2d"
    if q_scales.dim() != 2 or k_scales.dim() != 2:
        return a, False, "qk_scales_not_2d"
    has_v = v_qweight is not None and v_scales is not None
    if has_v and (v_qweight.dim() != 2 or v_scales.dim() != 2):
        return a, False, "v_bad_shape"
    if int(group_size) != 32:
        return a, False, "group_size_not_32"
    k = int(a.shape[1])
    if k % 32 != 0:
        return a, False, "k_not_group_aligned"
    if k != int(q_qweight.shape[1]) * 8 or k != int(k_qweight.shape[1]) * 8:
        return a, False, "qk_k_mismatch"
    if has_v and k != int(v_qweight.shape[1]) * 8:
        return a, False, "v_k_mismatch"
    dev = a.device
    tensors = [q_qweight, k_qweight, q_scales, k_scales]
    if has_v:
        tensors.extend([v_qweight, v_scales])  # type: ignore[arg-type]
    if any(t.device != dev for t in tensors):
        return a, False, "device_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    try:
        c = packed_int4_symmetric_fused_qkv_m1(
            a.contiguous(),
            q_qweight.contiguous(),
            k_qweight.contiguous(),
            v_qweight.contiguous() if has_v else None,
            q_scales.contiguous(),
            k_scales.contiguous(),
            v_scales.contiguous() if has_v else None,
            int(group_size),
            config=config,
            policy=policy,
        )
        expected_n = (
            int(q_qweight.shape[0])
            + int(k_qweight.shape[0])
            + (int(v_qweight.shape[0]) if has_v else 0)
        )
        if c.shape != (1, expected_n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"


def packed_int4_symmetric_fused_gate_up_m1_safe(
    a: torch.Tensor,
    gate_qweight: torch.Tensor,
    up_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    up_scales: torch.Tensor,
    group_size: int,
    *,
    activation: str = "silu",
    out: torch.Tensor | None = None,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if not awq_fused_gate_up_enabled(config=config, policy=policy):
        return a, False, "disabled"
    if a.dim() != 2 or int(a.shape[0]) != 1:
        return a, False, "input_not_m1_2d"
    if gate_qweight.dim() != 2 or up_qweight.dim() != 2:
        return a, False, "qweight_not_2d"
    if gate_scales.dim() != 2 or up_scales.dim() != 2:
        return a, False, "scales_not_2d"
    if tuple(gate_qweight.shape) != tuple(up_qweight.shape):
        return a, False, "qweight_shape_mismatch"
    if tuple(gate_scales.shape) != tuple(up_scales.shape):
        return a, False, "scale_shape_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    if int(group_size) != 32:
        return a, False, "group_size_not_32"
    if int(a.shape[1]) != int(gate_qweight.shape[1]) * 8:
        return a, False, "k_mismatch"
    if int(a.shape[1]) % 32 != 0:
        return a, False, "k_not_group_aligned"
    dev = a.device
    if (
        gate_qweight.device != dev
        or up_qweight.device != dev
        or gate_scales.device != dev
        or up_scales.device != dev
    ):
        return a, False, "device_mismatch"
    try:
        c = packed_int4_symmetric_fused_gate_up_m1(
            a.contiguous(),
            gate_qweight.contiguous(),
            up_qweight.contiguous(),
            gate_scales.contiguous(),
            up_scales.contiguous(),
            int(group_size),
            activation=activation,
            out=out,
            config=config,
            policy=policy,
        )
        if c.shape != (1, int(gate_qweight.shape[0])):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"


def awq_fused_gemm_safe(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if a.dim() != 2:
        return a, False, "input_not_2d"
    if qweight.dim() != 2:
        return a, False, "qweight_not_2d"
    if scales.dim() != 2:
        return a, False, "scales_not_2d"
    if qzeros is None or qzeros.dim() != 2:
        return a, False, "qzeros_bad"
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(qweight.shape[0])
    if k != int(qweight.shape[1] * 8):
        return a, False, "k_mismatch"
    if group_size <= 0 or k % int(group_size) != 0:
        return a, False, "group_mismatch"
    if (
        a.device != qweight.device
        or a.device != scales.device
        or a.device != qzeros.device
    ):
        return a, False, "device_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    if bias is not None:
        if bias.dim() != 1 or int(bias.numel()) != n:
            return a, False, "bias_bad_shape"
        if bias.device != a.device:
            return a, False, "bias_device_mismatch"
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return a, False, f"unsupported_bias_dtype_{str(bias.dtype)}"
    a = a.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    bias_arg = bias.contiguous() if bias is not None else None
    try:
        c = awq_fused_gemm(
            a,
            qweight,
            scales,
            qzeros,
            group_size=group_size,
            out=out,
            bias=bias_arg,
            config=config,
            policy=policy,
        )
        if c.shape != (m, n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"


def packed_int4_symmetric_fused_gemm_safe(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if a.dim() != 2:
        return a, False, "input_not_2d"
    if qweight.dim() != 2:
        return a, False, "qweight_not_2d"
    if scales.dim() != 2:
        return a, False, "scales_not_2d"
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(qweight.shape[0])
    if k != int(qweight.shape[1] * 8):
        return a, False, "k_mismatch"
    if group_size <= 0 or k % int(group_size) != 0:
        return a, False, "group_mismatch"
    if a.device != qweight.device or a.device != scales.device:
        return a, False, "device_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    if bias is not None:
        if bias.dim() != 1 or int(bias.numel()) != n:
            return a, False, "bias_bad_shape"
        if bias.device != a.device:
            return a, False, "bias_device_mismatch"
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return a, False, f"unsupported_bias_dtype_{str(bias.dtype)}"
    a = a.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    bias_arg = bias.contiguous() if bias is not None else None
    try:
        c = packed_int4_symmetric_fused_gemm(
            a,
            qweight,
            scales,
            group_size=group_size,
            out=out,
            bias=bias_arg,
            config=config,
            policy=policy,
        )
        if c.shape != (m, n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"
