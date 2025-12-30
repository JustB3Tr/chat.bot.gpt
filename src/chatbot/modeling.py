from transformers import GPT2Config, GPT2LMHeadModel

from .config import ModelConfig


def build_model(config: ModelConfig) -> GPT2LMHeadModel:
    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_ctx=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        resid_pdrop=config.dropout,
        embd_pdrop=config.dropout,
        attn_pdrop=config.dropout,
        layer_norm_epsilon=1e-5,
    )
    return GPT2LMHeadModel(model_config)
