# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, MutableMapping
from pathlib import Path

import aiohttp
import requests
from urllib3.util import parse_url

from vllm.version import __version__ as VLLM_VERSION

class HTTPConnection:
The global [`HTTPConnection`][vllm.connections.HTTPConnection] instance used
by vLLM.
