# SPDX-License-Identifier: Apache-2.0
class Platform:
    def __init__(self):
        self.device_type = "cuda"
        self.supported_quantization = ["awq", "gptq", "fp8", "gguf"]
    def is_cuda(self): return True
    def is_rocm(self): return False
    def is_macos(self): return False

current_platform = Platform()
