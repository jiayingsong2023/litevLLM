# SPDX-License-Identifier: Apache-2.0

def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def next_power_of_2(n: int) -> int:
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)

def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y
