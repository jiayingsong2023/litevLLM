# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

@contextmanager
def record_function_or_nullcontext(*args, **kwargs):
    yield

def tensor_data(*args, **kwargs): pass
