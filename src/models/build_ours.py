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
from .build import CustomVLMModel
from .config import VisionLanguageConfig
from src.constant import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class CustomVLM(CustomVLMModel):
    """Vision–Language 모델 (Vision Encoder + Projector + LLM)."""

    config_class = VisionLanguageConfig

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    def __init__(self, config, **kwargs):
        # 부모 클래스 초기화 호출 - 이 줄이 없으면 부모 __init__은 실행되지 않음
        super().__init__(config, **kwargs)
        
        # 추가 초기화 코드
        self.vision_prompt = "새로운 속성"


    def _prepare_multimodal_inputs(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ):
        """forward와 generate에서 공통적으로 사용되는 입력 전처리 로직"""
        # 1. 토큰 전처리
        processed_input_ids = self.preprocess_image_tokens(input_ids)
        if labels is not None:
            processed_labels = self.preprocess_image_tokens(labels)
        else:
            processed_labels = processed_input_ids.clone()
        
        # 2. 비전 인코딩
        self.current_batch_size = processed_input_ids.size(0)
        v_embs = self._get_vision_embeds(pixel_values)  # (B*num_samples, N', d_l)
        
        # 3. v_embs 배치 처리
        v_embs = list(torch.split(v_embs, self.current_batch_size, dim=0))  # (B, num_samples, N', d_l)
        
        for i, v_emb in enumerate(v_embs):
            # 4. 비전 임베딩 풀링
            if self.config.mm_spatial_pool_mode != "none":
                v_emb = self._get_2dPool(v_emb, stride=2)
            # 5. 줄바꿈 토큰 삽입
            v_emb = self.newline_inserter(v_emb, self.image_newline)
            v_embs[i] = v_emb
        
        # 6. 이미지 토큰 대체
        inp_emb, pad_lbl, pad_mask, pos_ids = self._replace_image_tokens_with_features(
            input_ids=processed_input_ids,
            labels=processed_labels,
            attention_mask=attention_mask,
            image_features=v_embs,
            embed_tokens_fn=self.llm.get_input_embeddings(),
            image_token_index=IMAGE_TOKEN_INDEX,
            ignore_index=IGNORE_INDEX,
            max_length=self.config.language_config.max_position_embeddings,
            padding_side=self.tokenizer.padding_side,
        )
        
        return inp_emb, pad_lbl, pad_mask, pos_ids

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 공통 입력 전처리 메서드 호출
        inp_emb, pad_lbl, pad_mask, pos_ids = self._prepare_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # LLM 모델 호출
        return self.llm(
            inputs_embeds=inp_emb,
            attention_mask=pad_mask if attention_mask is not None else None,
            position_ids=pos_ids,
            labels=pad_lbl,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):
        # 공통 입력 전처리 메서드 호출
        inp_emb, _, pad_mask, pos_ids = self._prepare_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # generate에 필요한 model_kwargs 구성
        model_kwargs = {
            "inputs_embeds": inp_emb,
            "attention_mask": pad_mask if attention_mask is not None else None,
            "position_ids": pos_ids,
        }
        
        # 생성 수행
        outputs = self.llm.generate(
            input_ids=None,  # input_ids 대신 inputs_embeds 사용
            **model_kwargs,
            **generate_kwargs
        )
        
        return outputs
if __name__ == "__main__":
    vision = ["facebook/dinov2-small","facebook/dino-vitb16"]
    llm = ["gpt2","Qwen/Qwen3-0.6B","Qwen/Qwen3-4B",]
    cfg = VisionLanguageConfig(
        vision_model_name=vision[0],
        language_model_name=llm[1],
        use_resampler=False,
        mm_spatial_pool_mode= "average",
    )
    model = CustomVLM(cfg).eval().to("cpu")

    dummy_img = torch.randn(4, 3, 224, 224, device="cpu", dtype=torch.float16)
    prompt = f"{DEFAULT_IM_START_TOKEN} hello {DEFAULT_IMAGE_TOKEN} world {DEFAULT_IM_END_TOKEN}"
    # Create a batch of prompts
    prompts = [
        f"{DEFAULT_IM_START_TOKEN} hello {DEFAULT_IMAGE_TOKEN} world {DEFAULT_IM_END_TOKEN}",
        f"{DEFAULT_IM_START_TOKEN} test {DEFAULT_IMAGE_TOKEN} example {DEFAULT_IM_END_TOKEN}"
    ]
    tok = model.tokenizer(prompts, return_tensors="pt", padding=True).to("cpu")
    
    with torch.no_grad():
        out = model(pixel_values=dummy_img, input_ids=tok.input_ids, attention_mask=tok.attention_mask, labels=tok.input_ids)
        print("Logits shape:", out.logits.shape)
    with torch.no_grad():
        out = model.generate(pixel_values=dummy_img, input_ids=tok.input_ids,max_new_tokens =50 ,attention_mask=tok.attention_mask, labels=tok.input_ids)
        print("Generated IDs:", out)
        out = model.tokenizer.batch_decode(out, skip_special_tokens=True)
        print("Generated text:", out)

    
