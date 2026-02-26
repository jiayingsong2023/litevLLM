# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    return -(a // -b)

def next_power_of_2(n: int) -> int:
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)

def round_up(x: int, y: int) -> int:
    return (x // y) * y
