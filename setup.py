import os
from setuptools import setup, find_packages
from setuptools_scm import get_version

def get_vllm_version() -> str:
    try:
        return get_version(write_to="vllm/_version.py")
    except Exception:
        return "0.16.0"

# litevLLM - Simplified setup.py for Triton/Python Only
setup(
    name="vllm",
    version=get_vllm_version(),
    packages=find_packages(exclude=("tests", "benchmarks", "csrc", "cmake")),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "triton",
        "transformers",
        "safetensors",
        "sentencepiece",
        "numpy",
        "requests",
        "tqdm",
        "pyyaml",
        "pillow",
        "ray>=2.9.0",
        "prometheus_client",
        "ninja",
        "amdsmi",
    ],
    entry_points={
        "console_scripts": [
            "vllm=vllm.entrypoints.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vllm": ["*.json", "*.jinja", "*.txt"],
    },
)
