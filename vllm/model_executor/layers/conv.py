# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    num_dim: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | Literal["same", "valid"] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str) and padding not in valid_padding_strings:
            raise ValueError(
                f"Invalid padding string '{padding}'. "
                f"Expected one of {valid_padding_strings}."
            )

        if padding == "same":
            padding = (
                kernel_size // 2
                if isinstance(kernel_size, int)
                else tuple(k // 2 for k in kernel_size)
            )
        elif padding == "valid":
            padding = 0

        kernel_size = (
            (kernel_size,) * self.num_dim
            if isinstance(kernel_size, int)
            else kernel_size
        )
        stride = (stride,) * self.num_dim if isinstance(stride, int) else stride
        padding = (padding,) * self.num_dim if isinstance(padding, int) else padding
        dilation = (dilation,) * self.num_dim if isinstance(dilation, int) else dilation

        if padding == "same" and any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.enable_linear = (
            (self.kernel_size == self.stride)
            and not any(self.padding)
            and self.groups == 1
        )
        self.input_size = in_channels * math.prod(self.kernel_size)

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *kernel_size,
                dtype=params_dtype,
            ),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"bias={self.bias is not None}"
        return s

# --8<-- [start:conv2d]
@CustomOp.register("conv2d")
class Conv2dLayer(ConvLayerBase):
        assert x.dim() == 4
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # By default, we use CUDNN's convolution ops with optimization.
        return self._forward_conv(x)

class CausalConv2dLayer(Conv2dLayer):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        if padding is not None:
            raise ValueError(
                "Argument padding should be set to None for CausalConv2dLayer."
            )
        self._left_padding: int = kernel_size - 1
        self._right_padding: int = stride - 1
        padding = 0

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            params_dtype=params_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.pad(x, pad=(self._left_padding, self._right_padding, 0, 0))
        x = super().forward(x)
        return x

# --8<-- [start:conv3d]
@CustomOp.register("conv3d")
class Conv3dLayer(ConvLayerBase):
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch2.9.0 disabled CUDNN's Conv3D, which caused a
        # significant performance regression.
        # See: https://github.com/vllm-project/vllm/issues/27406
        # and https://github.com/pytorch/pytorch/issues/166122
        # By default, we use CUDNN's convolution ops with optimization.
        if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):
            return self._forward_mulmat(x)
        return self._forward_conv(x)
