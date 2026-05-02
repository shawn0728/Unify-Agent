# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Interleaved multi-modal inferencer for the Bagel model.

The :class:`InterleaveInferencer` exposes a small set of primitives
(``init_gen_context``, ``update_context_text``, ``update_context_image``,
``gen_text``, ``gen_image``) which can be composed into multi-turn
conversations that mix text and image tokens. It also offers two convenience
entry points:

* :meth:`InterleaveInferencer.interleave_inference` – run a single forward
  pass over an interleaved list of strings / PIL images.
* :meth:`InterleaveInferencer.__call__` – the standard image-edit / T2I
  shortcut used by the eval scripts.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer.\n"
    "The reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here"
)

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then "
    "generate the image.\n"
    "The planning process is enclosed within <think> </think> tags, i.e. "
    "<think> planning process here </think> image here"
)


class InterleaveInferencer:
    """Stateful wrapper around a Bagel model for interleaved generation."""

    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self) -> Dict[str, Any]:
        """Create an empty KV-cache context."""
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    @torch.no_grad()
    def update_context_text(self, text: str, gen_context: Dict[str, Any]) -> Dict[str, Any]:
        """Append a text segment to the cached context (single-batch only)."""
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_image(
        self,
        image: Image.Image,
        gen_context: Dict[str, Any],
        vae: bool = True,
        vit: bool = True,
    ) -> Dict[str, Any]:
        """Append an image segment, optionally encoded with both VAE and ViT."""
        assert vae or vit, "At least one of vae/vit encoders must be enabled"
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_text_precontext: Optional[Dict[str, Any]] = None,
        cfg_img_precontext: Optional[Dict[str, Any]] = None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        enable_taylorseer: bool = False,
    ):
        """Generate a single image from the current context using rectified-flow sampling."""
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_precontext["kv_lens"],
            curr_rope=cfg_text_precontext["ropes"],
            image_sizes=[image_shape],
        )

        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_precontext["kv_lens"],
            curr_rope=cfg_img_precontext["ropes"],
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
            enable_taylorseer=enable_taylorseer,
        )

        return self.decode_image(unpacked_latent[0], image_shape)

    def decode_image(self, latent: torch.Tensor, image_shape) -> Image.Image:
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1, h, w,
            self.model.latent_patch_size, self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    @torch.no_grad()
    def gen_text(
        self,
        gen_context: Dict[str, Any],
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        """Generate text tokens conditioned on the current context.

        The context is *deep-copied* so that callers can keep generating from
        the same upstream state without being affected by the rolling KV-cache
        produced during decoding.
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        raw_output = self.tokenizer.decode(
            unpacked_latent[:, 0], skip_special_tokens=False
        )
        # Strip leading/trailing chat tags when present; otherwise return as-is.
        try:
            output = raw_output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        except Exception:
            output = raw_output
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think: bool = False,
        understanding_output: bool = False,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval=(0.4, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes=(1024, 1024),
        enable_taylorseer: bool = False,
    ) -> List[Union[str, Image.Image]]:
        """Walk through an interleaved list of inputs and return the model outputs.

        The list is consumed left-to-right; strings extend the prompt context
        while ``PIL.Image`` instances are encoded with both VAE and ViT. After
        the inputs are absorbed, the model produces (optionally) a thinking
        text and then a final image, *unless* ``understanding_output`` is set,
        in which case only text is generated.
        """
        output_list: List[Union[str, Image.Image]] = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    rgb = pil_img2rgb(input_term)
                    resized = self.vae_transform.resize_transform(rgb)
                    gen_context = self.update_context_image(
                        pil_img2rgb(resized), gen_context, vae=True, vit=True
                    )
                    image_shapes = resized.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)
            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )
                output_list.append(img)

        return output_list

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience wrapper around :meth:`interleave_inference`.

        Returns a dict with ``image`` and ``text`` keys; either may be ``None``
        depending on the active mode.
        """
        output: Dict[str, Any] = {"image": None, "text": None}
        if image is None and text is None:
            print("Please provide at least one input: either an image or text.")
            return output

        input_list: List[Union[str, Image.Image]] = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        for item in self.interleave_inference(input_list, **kwargs):
            if isinstance(item, Image.Image):
                output["image"] = item
            elif isinstance(item, str):
                output["text"] = item
        return output
