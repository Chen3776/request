from __future__ import annotations
import enum
from dataclasses import dataclass

# Helper classes from the original configuration file
class StrEnum(str, enum.Enum):
    pass

class InitFnType(StrEnum):
    normal = "normal"
    mitchell = "mitchell"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"

class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class BlockType(StrEnum):
    sequential = "sequential"
    llama = "llama"

class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    gemma_rms = "gemma_rms"

class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    fine_grained = "fine_grained"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"

# A simplified ModelConfig based on what LLaDABlock requires
@dataclass
class ModelConfig:
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    mlp_ratio: int = 4
    max_sequence_length: int = 2048
    vocab_size: int = 50257
    init_device: str = "cuda"
    init_fn: InitFnType = InitFnType.normal
    init_std: float = 0.02
    block_type: BlockType = BlockType.sequential
    activation_type: ActivationType = ActivationType.swiglu
    layer_norm_type: LayerNormType = LayerNormType.rms
    rms_norm_eps: float = 1e-5
    rope: bool = True
    rope_theta: float = 10000.0
    rope_full_precision: bool = True
    flash_attention: bool = False # This will be toggled by the sdp_kernel
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    include_bias: bool = False
    
    # Defaults for other fields that might be accessed
    init_cutoff_factor: float | None = None
    embedding_size: int | None = None
    block_group_size: int = 1
    alibi: bool = False
    alibi_bias_max: int = 8
    layer_norm_with_affine: bool = True
    bias_for_layer_norm: bool | None = None
    attention_layer_norm: bool = False
    attention_layer_norm_with_affine: bool = False
    effective_n_kv_heads: int | None = None
    mlp_hidden_size: int | None = None
    include_qkv_bias: bool = False
    input_emb_norm: bool = False
    scale_logits: bool = False
    weight_tying: bool = False

    def __post_init__(self):
        if self.effective_n_kv_heads is None:
            self.effective_n_kv_heads = self.n_heads
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = self.mlp_ratio * self.d_model


# A simplified LLaDAConfig for the wrapper model
@dataclass
class LLaDAConfig(ModelConfig):
    # This class inherits all fields from ModelConfig and can be extended
    # For this test, it can be the same as ModelConfig
    use_cache: bool = True
    use_return_dict: bool = True