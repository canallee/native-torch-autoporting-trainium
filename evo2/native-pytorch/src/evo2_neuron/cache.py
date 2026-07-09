"""Inference-time caches for Evo2 blocks.

StripedHyena2 has four block types with different caching needs:

  * `attn` blocks   -> InferenceParams (standard KV cache)
  * `hcl` blocks    -> HyenaCascadeIIRInferenceParams (FIR window + IIR state)
  * `hcm` blocks    -> HyenaCascadeFIRInferenceParams (outer FIR + inner FIR)
  * `hcs` blocks    -> HyenaCascadeFIRInferenceParams (outer FIR + inner FIR)

Layer outputs of these caches are wrapped together inside an HF Cache subclass
(`Evo2Cache`) so model.generate() can drive autoregressive decoding without
the user having to instantiate four separate caches by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class InferenceParams:
    """Standard KV cache for attention blocks."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class HyenaCascadeIIRInferenceParams:
    """Cache for `hcl` blocks: short FIR window + IIR modal state."""

    fir_filter_length: int = 3
    state_dim: int = 16
    seqlen_offset: int = 0
    fir_state_dict: dict = field(default_factory=dict)
    state_dict: dict = field(default_factory=dict)

    def reset(self):
        self.seqlen_offset = 0


@dataclass
class HyenaCascadeFIRInferenceParams:
    """Cache for `hcm` and `hcs` blocks: outer short FIR + inner FIR cascade."""

    fir_filter_length: int = 3
    fir_inner_filter_length: int = 4
    seqlen_offset: int = 0
    fir_inner_state_dict: dict = field(default_factory=dict)
    fir_state_dict: dict = field(default_factory=dict)
    state_dict: dict = field(default_factory=dict)

    def reset(self):
        self.seqlen_offset = 0


class Evo2Cache:
    """Container for per-block-type inference params.

    Not a transformers.Cache subclass (the new Cache API requires per-layer
    dataclasses, which doesn't fit StripedHyena 2's 4 block-type-specific
    state structures). Instead we set Evo2PreTrainedModel._supports_cache_class
    = False so HF's generate() treats this as an opaque past_key_values dict.
    """

    is_compileable = False

    def __init__(
        self,
        max_seqlen: int,
        max_batch_size: int,
        short_filter_length: int,
        hcm_filter_length: int,
        hcs_filter_length: int,
        state_size: int,
    ):
        self.mha = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=max_batch_size,
        )
        self.hcl = HyenaCascadeIIRInferenceParams(
            fir_filter_length=short_filter_length,
            state_dim=state_size,
        )
        self.hcm = HyenaCascadeFIRInferenceParams(
            fir_filter_length=short_filter_length,
            fir_inner_filter_length=hcm_filter_length,
        )
        self.hcs = HyenaCascadeFIRInferenceParams(
            fir_filter_length=short_filter_length,
            fir_inner_filter_length=hcs_filter_length,
        )

    @property
    def seqlen_offset(self) -> int:
        return self.mha.seqlen_offset

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.mha.seqlen_offset

    def get_max_cache_shape(self) -> int:
        return self.mha.max_seqlen

    def get_max_length(self) -> int:
        return self.mha.max_seqlen

    def advance(self, n: int = 1) -> None:
        self.mha.seqlen_offset += n
        self.hcl.seqlen_offset += n
        self.hcm.seqlen_offset += n
        self.hcs.seqlen_offset += n

    def set_offset(self, offset: int) -> None:
        self.mha.seqlen_offset = offset
        self.hcl.seqlen_offset = offset
        self.hcm.seqlen_offset = offset
        self.hcs.seqlen_offset = offset

    def reset(self) -> None:
        self.mha.reset(self.mha.max_seqlen, self.mha.max_batch_size)
        self.hcl.reset()
        self.hcm.reset()
        self.hcs.reset()

    def by_block_name(self, name: str):
        return getattr(self, name)
