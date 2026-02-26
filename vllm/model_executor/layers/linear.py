# SPDX-License-Identifier: Apache-2.0

from .lite_linear import LiteLinear

# Alias all parallel layers to LiteLinear
# LiteLinear handles the differences via its flexible __init__
ColumnParallelLinear = LiteLinear
RowParallelLinear = LiteLinear
QKVParallelLinear = LiteLinear
MergedColumnParallelLinear = LiteLinear
ReplicatedLinear = LiteLinear

# Keep LinearBase/LinearMethodBase as dummy or import real ones if needed elsewhere
# But mostly models just import *Linear classes.
class LinearBase: pass
class LinearMethodBase: pass
class UnquantizedLinearMethod: pass # stub, handled inside LiteLinear
