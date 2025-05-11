from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from config import VisionLanguageConfig
from projector import build_vision_projector
from newline import NewlineTokenInserter
from constant import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,  # placeholder, runtime에는 dynamic ID 사용
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class CustomVLMModel(PreTrainedModel):
    """Vision–Language 모델 (Vision Encoder + Projector + LLM).

    Special‑token ID 는 HF Tokenizer 가 런타임에 할당하므로, 실제 학습/추론 단계에서는
    `self.special_token_ids["<image>"]` 형태로 접근한다.
    """

    config_class = VisionLanguageConfig

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: VisionLanguageConfig,
        *,
        vision_dtype: torch.dtype = torch.float16,
        llm_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(config)

        # 1) Vision encoder --------------------------------------------------
        self.vision_encoder = AutoModel.from_pretrained(
            config.vision_model_name,
            torch_dtype=vision_dtype,
        )
        d_v = self.config.vision_config.hidden_size

        # 2) Language model --------------------------------------------------
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_name,
            torch_dtype=llm_dtype,
        )
        d_l = self.config.language_config.hidden_size

        # 3) Projector -------------------------------------------------------
        self.projector = build_vision_projector(
            d_v=d_v,
            d_l=d_l,
            projector_type=config.projector_type,
            vision_cfg=self.config.vision_config,
        ).to(llm_dtype)

        # 4) Optional resampler ---------------------------------------------
        if getattr(config, "use_resampler", False):
            from resampler.mamba_ssm.modules.mamba_compressor import MambaCompressor

            self.resampler = MambaCompressor(d_model=d_v, n_layer=1, fp32=False)

        # 5) NEWLINE token parameter ----------------------------------------
        self.image_newline = nn.Parameter(
            torch.zeros(self.config.language_config.hidden_size, dtype=llm_dtype)
        )
        self.newline_inserter = NewlineTokenInserter(config)

        # 6) Tokenizer & special tokens -------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
        if self.tokenizer.pad_token is None:
            # GPT‑계열은 pad token 없음 → eos token 재사용
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        self._register_special_tokens()

        # 7) Optional freezing ----------------------------------------------
        if getattr(config, "freeze_vision", True):
            self._freeze(self.vision_encoder)
        if getattr(config, "freeze_llm", False):
            self._freeze(self.llm)

        self.post_init()

    # ------------------------------------------------------------------ #
    # SPECIAL‑TOKEN HANDLING
    # ------------------------------------------------------------------ #
    def _register_special_tokens(self):
        """Tokenizer 에 special tokens 등록 후 dynamic ID 저장."""
        special_tokens = {
            "additional_special_tokens": [
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
            ]
        }
        added = self.tokenizer.add_special_tokens(special_tokens)
        if added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.special_token_ids = {
            DEFAULT_IMAGE_TOKEN: self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN),
            DEFAULT_IMAGE_PATCH_TOKEN: self.tokenizer.convert_tokens_to_ids(
                DEFAULT_IMAGE_PATCH_TOKEN
            ),
            DEFAULT_IM_START_TOKEN: self.tokenizer.convert_tokens_to_ids(
                DEFAULT_IM_START_TOKEN
            ),
            DEFAULT_IM_END_TOKEN: self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN),
        }

    # ------------------------------------------------------------------ #
    # PRIVATE UTILITIES
    # ------------------------------------------------------------------ #
    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # VISION → (Optional) RESAMPLER
    # ------------------------------------------------------------------ #
    def _get_vision_embeds(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Vision encoder → (optional) resampler → projector → LLM‑space embeds."""
        v_feat = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  # (B,N,d_v)

        if getattr(self.config, "use_resampler", False):
            v_feat = self._get_2dPool(v_feat)  # 차원 유지 (B,N',d_v)

        v_emb = self.projector(v_feat)  # (B,N',d_l)
        return v_emb

    # ------------------------------------------------------------------ #
    # 2‑D Spatial/Temporal Pooling (for Resampler)
    # ------------------------------------------------------------------ #
    def _get_2dPool(self, feature: torch.FloatTensor, stride: int = 2):
        """ViT patch token을 2‑D grid 로 reshape 후 pooling."""
        # feature: (B, N, d_v)
        num_frames, num_tokens, num_dim = feature.shape
        side = int(math.sqrt(num_tokens))
        assert side * side == num_tokens, "토큰 수가 완전한 정사각형이 아님"
        print(f"feature shape: {feature.shape}, side: {side}")
        feature = feature.view(num_frames, side, side, num_dim)  # (B, H, W, d_v)

        # (Optional) temporal pooling (영상 입력 대응)
        temporal_pooling = getattr(self.config, "temporal_pooling", 1)
        if temporal_pooling > 1:
            feature = feature.reshape(num_frames, num_tokens, num_dim)
            feature = feature.permute(1, 2, 0)  # (N, d_v, B)
            feature = nn.functional.avg_pool1d(feature, kernel_size=temporal_pooling, stride=temporal_pooling)
            feature = feature.permute(2, 0, 1)
            num_frames //= temporal_pooling
            feature = feature.view(num_frames, side, side, num_dim)

        # (B, H, W, d_v) → (B, d_v, H, W)
        feature = feature.permute(0, 3, 1, 2).contiguous()

        mode = getattr(self.config, "mm_spatial_pool_mode", "average")
        if mode == "average":
            feature = nn.functional.avg_pool2d(feature, stride)
        elif mode == "max":
            feature = nn.functional.max_pool2d(feature, stride)
        elif mode == "bilinear":
            height, width = feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            feature = nn.functional.interpolate(feature, size=scaled_shape, mode="bilinear")
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {mode}")
        print(f"feature shape after pooling: {feature.shape}")
        # (B, d_v, H', W') → (B, H'*W', d_v)
        feature = feature.permute(0, 2, 3, 1).contiguous().view(num_frames, -1, num_dim)
        return feature

    # ------------------------------------------------------------------ #
    # 이미지 토큰 치환 util
    # ------------------------------------------------------------------ #
    def _replace_image_tokens_with_features(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        image_features: torch.FloatTensor,  # (B, N', d_l)
        embed_tokens_fn: nn.Module,
        image_token_index: int,
        *,
        ignore_index: int = IGNORE_INDEX,
        max_length: int = None,
        padding_side: str = "right",
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """텍스트 안의 <image> 토큰을 실제 비전 임베딩으로 치환."""
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # Pad 제외하고 flatten list 로 분리
        input_ids_list = [cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)]
        labels_list = [cur_lbl[cur_mask] for cur_lbl, cur_mask in zip(labels, attention_mask)]

        new_input_embeds, new_labels = [], []

        for b_idx, cur_input_ids in enumerate(input_ids_list):
            cur_labels = labels_list[b_idx]
            num_images = (cur_input_ids == image_token_index).sum().item()

            if num_images == 0:
                cur_input_embeds = embed_tokens_fn(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                continue

            # split indices (텍스트 세그먼트 추출)
            split_points = [-1] + torch.where(cur_input_ids == image_token_index)[0].tolist() + [cur_input_ids.shape[0]]
            text_segments = [cur_input_ids[split_points[i] + 1 : split_points[i + 1]] for i in range(len(split_points) - 1)]
            label_segments = [cur_labels[split_points[i] + 1 : split_points[i + 1]] for i in range(len(split_points) - 1)]
            split_sizes = [seg.shape[0] for seg in text_segments]

            text_embeds = embed_tokens_fn(torch.cat(text_segments))
            text_embeds = torch.split(text_embeds, split_sizes, dim=0)

            seq_embeds, seq_labels = [], []
            for seg_idx in range(num_images + 1):
                seq_embeds.append(text_embeds[seg_idx])
                seq_labels.append(label_segments[seg_idx])
                if seg_idx < num_images:
                    vis_embed = image_features[b_idx]  # 동일 배치 index 사용 (B 차원)
                    seq_embeds.append(vis_embed)
                    seq_labels.append(
                        torch.full((vis_embed.shape[0],), ignore_index, dtype=labels.dtype, device=device)
                    )

            seq_embeds = torch.cat(seq_embeds)
            seq_labels = torch.cat(seq_labels)
            new_input_embeds.append(seq_embeds)
            new_labels.append(seq_labels)

        # ---------------------------------------------- Padding to max len
        if max_length is not None:
            new_input_embeds = [x[:max_length] for x in new_input_embeds]
            new_labels = [x[:max_length] for x in new_labels]

        max_seq_len = max(x.shape[0] for x in new_input_embeds)
        embed_dim = new_input_embeds[0].shape[1]

        padded_embeds = []
        new_labels_padded = torch.full((batch_size, max_seq_len), ignore_index, dtype=labels.dtype, device=device)
        new_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        new_position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

        for i, (embeds, lbls) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = embeds.shape[0]
            pad_len = max_seq_len - cur_len
            if padding_side == "left":
                pad_tensor = torch.zeros((pad_len, embed_dim), dtype=embeds.dtype, device=device)
                embeds = torch.cat([pad_tensor, embeds], dim=0)
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = lbls
                    new_attention_mask[i, -cur_len:] = 1
                    new_position_ids[i, -cur_len:] = torch.arange(cur_len, device=device)
            else:  # right pad
                pad_tensor = torch.zeros((pad_len, embed_dim), dtype=embeds.dtype, device=device)
                embeds = torch.cat([embeds, pad_tensor], dim=0)
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = lbls
                    new_attention_mask[i, :cur_len] = 1
                    new_position_ids[i, :cur_len] = torch.arange(cur_len, device=device)
            padded_embeds.append(embeds)

        new_input_embeds = torch.stack(padded_embeds, dim=0)  # (B, L, d)
        return new_input_embeds, new_labels_padded, new_attention_mask, new_position_ids

    # ------------------------------------------------------------------ #
    # FORWARD
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 1) Vision → projector (+ optional resampler inside)
        v_emb = self._get_vision_embeds(pixel_values)  # (B, N', d_l)

        # 2) NEWLINE token
        v_emb = self.newline_inserter(v_emb, self.image_newline)

        # 3) Replace <image> tokens
        inputs_embeds, labels_pad, attn_mask, pos_ids = self._replace_image_tokens_with_features(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_features=v_emb,
            embed_tokens_fn=self.llm.get_input_embeddings(),
            image_token_index=self.special_token_ids[DEFAULT_IMAGE_TOKEN],
            ignore_index=IGNORE_INDEX,
            max_length=self.config.language_config.max_position_embeddings,
            padding_side=self.tokenizer.padding_side,
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            labels=labels_pad,
            return_dict=True,
        )


if __name__ == "__main__":
    # quick smoke‑test
    cfg = VisionLanguageConfig(
        vision_model_name="google/vit-base-patch16-224",
        language_model_name="gpt2",
        use_resampler=True,
        
    )
    model = CustomVLMModel(cfg).eval().to("cuda")

    dummy_img = torch.randn(10, 3, 224, 224, device="cuda", dtype=torch.float16)
    prompt = f"{DEFAULT_IM_START_TOKEN} hello world {DEFAULT_IM_END_TOKEN}"
    tok = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    with torch.no_grad():
        out = model(pixel_values=dummy_img, input_ids=tok.input_ids, attention_mask=tok.attention_mask, labels=tok.input_ids)
    print("Logits shape:", out.logits.shape)
