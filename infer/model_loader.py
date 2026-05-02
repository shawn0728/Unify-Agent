# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Model loader utility for Unify-Agent inference.

Builds an :class:`InterleaveInferencer` around a Bagel checkpoint. The function
mirrors the loader used during SFT training and supports full bf16, NF4 and
INT8 weight loading paths.

Module layout note
------------------
The training-time packages (``data``, ``modeling``) live under the ``SFT/``
sub-directory of the Unify-Agent repository. To keep the inference scripts
import-compatible with the upstream Bagel layout (``from data.data_utils
import ...``), this module prepends ``<repo>/SFT`` and ``<repo>/infer`` to
``sys.path`` at import time so the SFT code path becomes importable as
top-level packages. ``SFT/`` is placed *first* so ``data`` / ``modeling``
always resolve to the training-time implementations regardless of any
shadow packages that may live next to the inference scripts.
"""

import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SFT_DIR = os.path.join(_REPO_ROOT, "SFT")

# Insert in reverse-priority order so that after both inserts ``sys.path``
# starts with ``[_SFT_DIR, _HERE, ...]``: ``SFT/`` wins for ``data`` /
# ``modeling`` imports, and ``infer/`` is visible so ``inferencer.py`` can be
# located from sibling scripts (e.g. ``eval/``).
for _p in (_HERE, _SFT_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from safetensors.torch import (
    load_file as load_safetensors_file,
    save_file as save_safetensors_file,
)

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


# Bagel-7B-MoT base vocabulary size. The training-time tokenizer adds a small
# number of structured-control tokens (<tool_call>, <recaption>, ...) on top of
# the Qwen2 tokenizer; the embedding matrix is sized for the full 152064 rows
# so that any unused rows act as reserved tokens.
_EXPECTED_VOCAB_SIZE = 152064


def _get_ema_vocab_size_from_header(ema_path: str):
    """Cheaply read the embedding row count from a safetensors header."""

    key = "language_model.model.embed_tokens.weight"
    try:
        with open(ema_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        shape = header.get(key, {}).get("shape")
        if isinstance(shape, list) and len(shape) >= 1:
            return int(shape[0])
    except Exception:
        return None
    return None


def _maybe_cast_ema_to_bfloat16(ema_path: str, cache_path: str = None) -> str:
    """Convert a (possibly fp32) EMA checkpoint to bf16 with on-disk caching.

    The cache is reused as long as it is newer than the source checkpoint.
    """

    if cache_path is None:
        stem, ext = os.path.splitext(ema_path)
        cache_path = f"{stem}.bf16{ext}"

    needs_convert = True
    if os.path.exists(cache_path):
        needs_convert = os.path.getmtime(cache_path) < os.path.getmtime(ema_path)

    if needs_convert:
        print(f"[model_loader] Converting EMA to bfloat16: {ema_path} -> {cache_path}")
        state = load_safetensors_file(ema_path, device="cpu")
        bf16_state = {
            k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
            for k, v in state.items()
        }
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        save_safetensors_file(bf16_state, cache_path)
        print("[model_loader] EMA bfloat16 cache saved.")
    else:
        print(f"[model_loader] Reusing EMA bfloat16 cache: {cache_path}")

    return cache_path


def load_model_and_inferencer(
    model_path: str,
    mode: int = 1,
    max_memory: dict = None,
    base_model_path: str = None,
    ema_path: str = None,
    cast_ema_to_bfloat16: bool = False,
    ema_bf16_cache_path: str = None,
):
    """Load a Bagel checkpoint and wrap it in an :class:`InterleaveInferencer`.

    Args:
        model_path: Directory containing the model artefacts (config files, the
            VAE, the tokenizer and ``ema.safetensors``). When ``base_model_path``
            and / or ``ema_path`` are provided they override the defaults.
        mode: 1=full bf16, 2=NF4 4-bit quantization, 3=INT8 quantization.
        max_memory: Optional ``accelerate`` ``max_memory`` map. Defaults to
            ``"80GiB"`` per visible CUDA device.
        base_model_path: Directory holding ``llm_config.json``,
            ``vit_config.json``, ``ae.safetensors`` and the tokenizer files.
            Defaults to ``model_path``.
        ema_path: Path to the safetensors weights to load. Defaults to
            ``<model_path>/ema.safetensors``.
        cast_ema_to_bfloat16: If True and the EMA weights are not yet bf16,
            convert them on disk (with caching) before loading.
        ema_bf16_cache_path: Optional override for the bf16 cache location.

    Returns:
        An :class:`InterleaveInferencer` ready for multi-modal inference.
    """

    if max_memory is None:
        device_count = max(1, torch.cuda.device_count())
        max_memory = {i: "80GiB" for i in range(device_count)}
    print(f"[model_loader] module_file={__file__}")

    config_model_path = base_model_path or model_path
    resolved_ema_path = ema_path or os.path.join(model_path, "ema.safetensors")

    if not os.path.exists(config_model_path):
        raise FileNotFoundError(f"Base model path not found: {config_model_path}")
    if not os.path.exists(resolved_ema_path):
        raise FileNotFoundError(f"EMA checkpoint not found: {resolved_ema_path}")

    if cast_ema_to_bfloat16:
        resolved_ema_path = _maybe_cast_ema_to_bfloat16(
            resolved_ema_path, ema_bf16_cache_path
        )
    print(f"[model_loader] resolved_ema_path={resolved_ema_path}")

    tokenizer = Qwen2Tokenizer.from_pretrained(config_model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    tokenizer_vocab_size = len(tokenizer)
    print(f"[model_loader] tokenizer_vocab_size={tokenizer_vocab_size}")

    llm_config = Qwen2Config.from_json_file(
        os.path.join(config_model_path, "llm_config.json")
    )
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    if llm_config.vocab_size != _EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "Bagel config vocab size mismatch: "
            f"config={llm_config.vocab_size}, expected={_EXPECTED_VOCAB_SIZE}."
        )

    ema_vocab_size = _get_ema_vocab_size_from_header(resolved_ema_path)
    if ema_vocab_size is not None and ema_vocab_size != _EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "EMA vocab size mismatch: "
            f"ema={ema_vocab_size}, expected={_EXPECTED_VOCAB_SIZE}. "
            "Please use a checkpoint trained with the same vocab size."
        )

    if tokenizer_vocab_size > _EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "Tokenizer vocab is larger than model vocab; token ids may overflow: "
            f"tokenizer={tokenizer_vocab_size}, model={_EXPECTED_VOCAB_SIZE}. "
            "Please use the exact tokenizer/config used during training."
        )
    if tokenizer_vocab_size < _EXPECTED_VOCAB_SIZE:
        print(
            f"[model_loader] tokenizer_vocab_size ({tokenizer_vocab_size}) < "
            f"model_vocab_size ({_EXPECTED_VOCAB_SIZE}); extra rows are kept as reserved tokens."
        )

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(config_model_path, "vit_config.json")
    )
    vit_config.rope = False
    # The last hidden layer is dropped to align with the SFT recipe.
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(
        local_path=os.path.join(config_model_path, "ae.safetensors")
    )

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() <= 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    if mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=resolved_ema_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model = load_and_quantize_model(
            model,
            weights_location=resolved_ema_path,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True,
            torch_dtype=torch.float32,
        )
        model = load_and_quantize_model(
            model,
            weights_location=resolved_ema_path,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"Unsupported loading mode: {mode}")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    return inferencer
