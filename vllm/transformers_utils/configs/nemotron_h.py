# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    This is the configuration class to store the configuration of a
    [`NemotronHModel`]. It is used to instantiate a NemotronH model according
    to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to
    that of the NemotronH-v0.1 model.
    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the NemotronH model. Defines the number of
            different tokens that can be represented by the `inputs_ids`
            passed when calling [`NemotronHModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be
            tied. Note that this is only relevant if the model has an output
            word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 52):
            Number of hidden layers in the Transformer encoder.
        hybrid_override_pattern (`str`, *optional*, defaults to
            `"M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"`):
            The pattern of the hybrid model. The pattern is a string of
            characters where each character represents
            M: Mamba2, *: Attention, -: MLP
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the
            Transformer encoder.
        attention_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to
            implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use
            Multi Head Attention (MHA), if `num_key_value_heads=1` the model
            will use Multi Query Attention (MQA) otherwise GQA is used.
        mlp_hidden_act (`str`, *optional*, defaults to "relu2"):
            The non-linear activation function in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether or not residuals should be in `float32`. If set to `False`
            residuals will keep the same `dtype` as the rest of the model.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`,
            all logits will be calculated. If an integer value, only last
            `num_logits_to_keep` logits will be calculated.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*, defaults to None):
            Sliding window attention window size.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used
            with.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels.
            These are available only if `mamba-ssm` and `causal-conv1d`
            are installed, and the mamba modules are running on a CUDA device.
        ssm_state_size (`int`, *optional*, defaults to 128):
            The dimension of the mamba state space latents.
        mamba_num_heads (`int`, *optional*, defaults to 128):
            Number of heads in Mamba layers.
        mamba_n_groups (`int`, *optional*, defaults to 8):
            Number of groups in Mamba layers.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each Mamba head.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the mamba intermediate size.
        mamba_hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the Mamba layers.
        mamba_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum value for the time step in Mamba.
        mamba_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum value for the time step in Mamba.
        mamba_dt_limit (`tuple`, *optional*, defaults to (0.0, float("inf"))):
            Limits for the time step in Mamba.
        mamba_dt_init_floor (`float`, *optional*, defaults to 1e-4):
            Floor value for time step initialization in Mamba.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the convolution layer of the mamba mixer
            block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the input and output projections of the
            mamba mixer block.
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            Size of chunks for Mamba processing.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
            Whether to rescale the pre-normalization residual connections.
