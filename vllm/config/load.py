# SPDX-License-Identifier: Apache-2.0
class LoadConfig:
    def __init__(self, load_format: str = "auto", download_dir: str = None):
        self.load_format = load_format
        self.download_dir = download_dir