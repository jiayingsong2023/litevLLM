# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class PoolingParams:
    """Parameters for pooling operations."""
    truncate_prompt_tokens: Optional[int] = None
    use_activation: bool = False
    dimensions: Optional[int] = None
    step_tag_id: Optional[int] = None
    returned_token_ids: Optional[List[int]] = None
    
    task: Any = None
    requires_token_ids: bool = False
    skip_reading_prefix_cache: bool = False
    extra_kwargs: dict = field(default_factory=dict)
    output_kind: str = "final_only"

    def clone(self) -> "PoolingParams":
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=self.use_activation,
            dimensions=self.dimensions,
            step_tag_id=self.step_tag_id,
            returned_token_ids=self.returned_token_ids,
            task=self.task,
            requires_token_ids=self.requires_token_ids,
            skip_reading_prefix_cache=self.skip_reading_prefix_cache,
            extra_kwargs=self.extra_kwargs.copy(),
            output_kind=self.output_kind
        )
