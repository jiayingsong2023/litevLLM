# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
    [`NemotronModel`]. It is used to instantiate a Nemotron model
    according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Nemotron-8B.

    Configuration objects inherit from [`PretrainedConfig`] and can be
    used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Nemotron model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`NemotronModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the
            Transformer decoder.
        head_dim (`int`, *optional*):
            Projection weights dimension in multi-head attention. Set to
            hidden_size // num_attention_heads if None
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to
            implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use
            Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention
            (MQA) otherwise GQA is used. When converting a multi-head
            checkpoint to a GQA checkpoint, each group key and value
            head should be constructed by meanpooling all the original
            heads within that group. For more details checkout
            [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it
            is not specified, will default to `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the
            decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used
            with.
        initializer_range (`float`, *optional*, defaults to 0.0134):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`dict`, *optional*):
            The parameters of the RoPE embeddings. Expected contents:
                `rope_theta` (`float`): The base period of the RoPE embeddings.
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear',
                    'dynamic', 'yarn', 'longrope', 'llama3'], with 'default' being the
                    original RoPE implementation.
                `partial_rotary_factor` (`float`, *optional*, defaults to 0.5):
                    Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output
            projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj and down_proj layers in the MLP
            layers.

    ```python
    >>> from transformers import NemotronModel, NemotronConfig
    >>> # Initializing a Nemotron nemotron-15b style configuration
    >>> configuration = NemotronConfig()
    >>> # Initializing a model from the nemotron-15b style configuration
    >>> model = NemotronModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
        Validate the `rope_parameters` configuration.
