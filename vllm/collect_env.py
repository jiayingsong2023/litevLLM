# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
# code borrowed from https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py

import datetime
import locale
import os
import subprocess
import sys

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
from collections import namedtuple

import regex as re

from vllm.envs import environment_variables

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "pip_version",  # 'pip' or 'pip3'
        "pip_packages",
        "conda_packages",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
        "rocm_version",  # vllm specific field
        "vllm_version",  # vllm specific field
        "vllm_build_flags",  # vllm specific field
        "gpu_topo",  # vllm specific field
        "env_vars",
    ],
)

DEFAULT_CONDA_PATTERNS = {
    "torch",
    "numpy",
    "cudatoolkit",
    "soumith",
    "mkl",
    "magma",
    "triton",
    "optree",
    "nccl",
    "transformers",
    "zmq",
    "nvidia",
    "pynvml",
    "flashinfer-python",
    "helion",
}

DEFAULT_PIP_PATTERNS = {
    "torch",
    "numpy",
    "mypy",
    "flake8",
    "triton",
    "optree",
    "onnx",
    "nccl",
    "transformers",
    "zmq",
    "nvidia",
    "pynvml",
    "flashinfer-python",
    "helion",
}

def run(command):
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out

def run_and_parse_first_match(run_lambda, command, regex):
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]

def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_CONDA_PATTERNS
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, [conda, "list"])
    if out is None:
        return out

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#") and any(name in line for name in patterns)
    )

def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")

def get_clang_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )

def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")

def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")

def get_gpu_info(run_lambda):
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = " ({})".format(prop.gcnArchName)
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)

def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")

def get_cudnn_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "hipcc --version", r"HIP version: (\S+)"
    )

def get_vllm_version():
    from vllm import __version__, __version_tuple__

    if __version__ == "dev":
        return "N/A (dev)"
    version_str = __version_tuple__[-1]
    if isinstance(version_str, str) and version_str.startswith("g"):
        # it's a dev build
        if "." in version_str:
            # it's a dev build containing local changes
            git_sha = version_str.split(".")[0][1:]
            date = version_str.split(".")[-1][1:]
            return f"{__version__} (git sha: {git_sha}, date: {date})"
        else:
            # it's a dev build without local changes
            git_sha = version_str[1:]  # type: ignore
            return f"{__version__} (git sha: {git_sha})"
    return __version__

def summarize_vllm_build_flags():
    # This could be a static method if the flags are constant, or dynamic if you need to check environment variables, etc.
    return "CUDA Archs: {}; ROCm: {}".format(
        os.environ.get("TORCH_CUDA_ARCH_LIST", "Not Set"),
        "Enabled" if os.environ.get("ROCM_HOME") else "Disabled",
    )

def get_gpu_topo(run_lambda):
    output = None

    if get_platform() == "linux":
        output = run_and_read_all(run_lambda, "nvidia-smi topo -m")
        if output is None:
            output = run_and_read_all(run_lambda, "rocm-smi --showtopo")

    return output

# example outputs of CPU infos
#  * linux
#    Architecture:            x86_64
#      CPU op-mode(s):        32-bit, 64-bit
#      Address sizes:         46 bits physical, 48 bits virtual
#      Byte Order:            Little Endian
#    CPU(s):                  128
#      On-line CPU(s) list:   0-127
#    Vendor ID:               GenuineIntel
#      Model name:            Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#        CPU family:          6
#        Model:               106
#        Thread(s) per core:  2
#        Core(s) per socket:  32
#        Socket(s):           2
#        Stepping:            6
#        BogoMIPS:            5799.78
#        Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr
#                             sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl
#                             xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq monitor ssse3 fma cx16
#                             pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand
#                             hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced
#                             fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap
#                             avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1
#                             xsaves wbnoinvd ida arat avx512vbmi pku ospke avx512_vbmi2 gfni vaes vpclmulqdq
#                             avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid md_clear flush_l1d arch_capabilities
#    Virtualization features:
#      Hypervisor vendor:     KVM
#      Virtualization type:   full
#    Caches (sum of all):
#      L1d:                   3 MiB (64 instances)
#      L1i:                   2 MiB (64 instances)
#      L2:                    80 MiB (64 instances)
#      L3:                    108 MiB (2 instances)
#    NUMA:
#      NUMA node(s):          2
#      NUMA node0 CPU(s):     0-31,64-95
#      NUMA node1 CPU(s):     32-63,96-127
#    Vulnerabilities:
#      Itlb multihit:         Not affected
#      L1tf:                  Not affected
#      Mds:                   Not affected
#      Meltdown:              Not affected
#      Mmio stale data:       Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
#      Retbleed:              Not affected
#      Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
#      Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
#      Spectre v2:            Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
#      Srbds:                 Not affected
#      Tsx async abort:       Not affected
#  * win32
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU0
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142
#
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU1
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142

def get_cpu_info(run_lambda):
    rc, out, err = 0, "", ""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            "wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID, \
        CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE"
        )
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = "None"
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info

def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform

def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")

def get_windows_version(run_lambda):
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")
    findstr_cmd = os.path.join(system_root, "System32", "findstr")
    return run_and_read_all(
        run_lambda, "{} os get Caption | {} /v Caption".format(wmic_cmd, findstr_cmd)
    )

def get_lsb_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )

def check_release_file(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )

def get_os(run_lambda):
    from platform import machine

    platform = get_platform()

    if platform == "win32" or platform == "cygwin":
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "macOS {} ({})".format(version, machine())

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        return "{} ({})".format(platform, machine())

    # Unknown platform
    return platform

def get_python_platform():
    import platform

    return platform.platform()

def get_libc_version():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())

def is_uv_venv():
    if os.environ.get("UV"):
        return True
    pyvenv_cfg_path = os.path.join(sys.prefix, "pyvenv.cfg")
    if os.path.exists(pyvenv_cfg_path):
        with open(pyvenv_cfg_path, "r") as f:
            return any(line.startswith("uv = ") for line in f)
    return False

def get_pip_packages(run_lambda, patterns=None):
==============================
        System Info
==============================
OS                           : {os}
GCC version                  : {gcc_version}
Clang version                : {clang_version}
CMake version                : {cmake_version}
Libc version                 : {libc_version}

==============================
       PyTorch Info
==============================
PyTorch version              : {torch_version}
Is debug build               : {is_debug_build}
CUDA used to build PyTorch   : {cuda_compiled_version}
ROCM used to build PyTorch   : {hip_compiled_version}

==============================
      Python Environment
==============================
Python version               : {python_version}
Python platform              : {python_platform}

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : {is_cuda_available}
CUDA runtime version         : {cuda_runtime_version}
CUDA_MODULE_LOADING set to   : {cuda_module_loading}
GPU models and configuration : {nvidia_gpu_models}
Nvidia driver version        : {nvidia_driver_version}
cuDNN version                : {cudnn_version}
HIP runtime version          : {hip_runtime_version}
MIOpen runtime version       : {miopen_runtime_version}
Is XNNPACK available         : {is_xnnpack_available}

==============================
          CPU Info
==============================
{cpu_info}

==============================
Versions of relevant libraries
==============================
{pip_packages}
{conda_packages}
==============================
         vLLM Info
==============================
ROCM Version                 : {rocm_version}
vLLM Version                 : {vllm_version}
vLLM Build Flags:
  {vllm_build_flags}
GPU Topology:
  {gpu_topo}

==============================
     Environment Variables
==============================
{env_vars}
