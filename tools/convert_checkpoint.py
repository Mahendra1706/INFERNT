import os
import sys
import json
import struct
import argparse
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Hugging Face checkpoint to INFERNT format")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Hugging Face model ID or local directory")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoint",
                        help="Target directory for converted files")
    return parser.parse_args()

def convert_config(hf_config, output_path):
    arch = "LlamaForCausalLM"
    if hasattr(hf_config, "architectures") and hf_config.architectures:
        arch = hf_config.architectures[0]

    config_dict = {
        "architecture": arch,
        "vocab_size": int(hf_config.vocab_size),
        "hidden_size": int(hf_config.hidden_size),
        "num_layers": int(hf_config.num_hidden_layers),
        "num_attention_heads": int(hf_config.num_attention_heads),
        "num_key_value_heads": int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)),
        "intermediate_size": int(hf_config.intermediate_size),
        "max_seq_len": int(getattr(hf_config, "max_position_embeddings", 2048)),
        "rms_norm_eps": float(getattr(hf_config, "rms_norm_eps", 1e-5)),
        "rope_theta": float(getattr(hf_config, "rope_theta", 10000.0))
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Exported config to {output_path}")
    return config_dict

def convert_tokenizer(tokenizer, output_path):
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    id_to_token = {v: k for k, v in vocab.items()}

    # Try to extract real SentencePiece scores from the underlying model
    sp_scores = {}
    try:
        if hasattr(tokenizer, 'sp_model'):
            sp_model = tokenizer.sp_model
            for i in range(sp_model.get_piece_size()):
                piece = sp_model.id_to_piece(i)
                score = sp_model.get_score(i)
                sp_scores[piece] = score
            print(f"  Extracted {len(sp_scores)} scores from SentencePiece model")
    except Exception as e:
        print(f"  Warning: Could not extract SentencePiece scores: {e}")

    tokens = []
    scores_list = []
    max_token_len = 0
    for i in range(vocab_size):
        piece = id_to_token.get(i, f"<token_{i}>")
        piece_bytes = piece.encode("utf-8")
        tokens.append(piece_bytes)
        if len(piece_bytes) > max_token_len:
            max_token_len = len(piece_bytes)

        # Use real SentencePiece score if available, otherwise use -i as fallback
        if piece in sp_scores:
            scores_list.append(sp_scores[piece])
        else:
            scores_list.append(-float(i))

    with open(output_path, "wb") as f:
        f.write(struct.pack("II", vocab_size, max_token_len))
        for i, piece_bytes in enumerate(tokens):
            score = scores_list[i]
            length = len(piece_bytes)
            f.write(struct.pack("fi", score, length))
            f.write(piece_bytes)

    print(f"Exported tokenizer ({vocab_size} tokens) to {output_path}")

def convert_weights(model, output_path):
    state_dict = model.state_dict()
    tensor_names = []

    if "model.embed_tokens.weight" in state_dict:
        tensor_names.append("model.embed_tokens.weight")

    num_layers = 0
    while f"model.layers.{num_layers}.input_layernorm.weight" in state_dict:
        num_layers += 1

    for i in range(num_layers):
        tensor_names.extend([
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        ])

    if "model.norm.weight" in state_dict:
        tensor_names.append("model.norm.weight")

    if "lm_head.weight" in state_dict:
        tensor_names.append("lm_head.weight")
    elif "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        tensor_names.append("lm_head.weight")

    with open(output_path, "wb") as f:
        magic = 0x494E4654
        version = 1
        tensor_count = len(tensor_names)
        f.write(struct.pack("III", magic, version, tensor_count))

        for name in tensor_names:
            tensor = state_dict[name].float().contiguous().cpu().numpy()
            name_bytes = name.encode("utf-8")
            name_len = len(name_bytes)
            dims = list(tensor.shape)
            num_dims = len(dims)
            num_elements = int(tensor.size)

            f.write(struct.pack("i", name_len))
            f.write(name_bytes)
            f.write(struct.pack("i", num_dims))
            for d in dims:
                f.write(struct.pack("i", int(d)))
            f.write(struct.pack("q", num_elements))
            f.write(tensor.tobytes())

    print(f"Exported {len(tensor_names)} tensors to {output_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading configuration from {args.model}...")
    hf_config = AutoConfig.from_pretrained(args.model)
    config_path = os.path.join(args.output_dir, "config.json")
    convert_config(hf_config, config_path)

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.bin")
    convert_tokenizer(tokenizer, tokenizer_path)

    print(f"Loading model weights from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    weights_path = os.path.join(args.output_dir, "model.bin")
    convert_weights(model, weights_path)

    print(f"Conversion completed successfully! Artifacts saved in {args.output_dir}")

if __name__ == "__main__":
    main()
