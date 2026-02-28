# SPDX-License-Identifier: Apache-2.0
import aiohttp
import requests

class HTTPConnection:
    """The global HTTPConnection instance used by vLLM."""
    def __init__(self):
        pass

global_http_connection = HTTPConnection()
