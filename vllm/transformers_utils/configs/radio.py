# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    This is the configuration class to store the configuration of a Radio
    vision model. It is used to instantiate a Radio model according to the
    specified arguments, defining the model architecture.

    Args:
        model_name: Name of the vision transformer model
            (e.g., "vit_base_patch16_224"). Used to determine architecture
            dimensions from `VIT_TIMM_DIM_BY_NAME`.
        image_size: The size (resolution) of each image.
        patch_size: The size (resolution) of each patch.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        qk_normalization: Whether to apply normalization to queries and keys.
        norm_type: The normalization type to use.
        layer_norm_eps: The epsilon used by the layer normalization layers.
        initializer_factor: A factor for initializing all weight matrices.
        hidden_act: The non-linear activation function in the encoder.
        cpe_max_size: Maximum image size for position embeddings.
        norm_mean: Mean values for image normalization (RGB channels).
            Defaults to (0.48145466, 0.4578275, 0.40821073)).
        norm_std: Standard deviation values for image normalization
            (RGB channels). Defaults to (0.26862954, 0.26130258, 0.27577711)).
        register_multiple: Number of register tokens to use.
        teachers: A list of teacher model configurations. Each teacher configuration is
            a dict with keys like "name" and some may have "use_summary".
        cls_token_per_teacher: Whether to use a separate CLS token for each teacher.
