from typing import Optional
import torch
from src.models.build import CustomVLMModel
from typing import Union, List

# 필요한 상수들을 직접 정의 (import 문제 해결)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class CaptioningVLM(CustomVLMModel):
    """Vision-Language model with automatic captioning of video features."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Additional initialization if needed
        self.system_instruction = "You are a helpful assistant."
        self.captioning_instruction = "Generate a short descriptive caption for this visual content."
        self.caption_prompt_template = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": self.captioning_instruction},
        ]
    
    @torch.no_grad()
    def _generate_captions_for_features(self, v_emb):
        """비주얼 특징에 대한 캡션 생성 - 단순화된 방식"""
        batch_size, seq_len, dim = v_emb.shape
        
        # 결과를 저장할 리스트들
        caption_embeds_list = []
        outputs_list = []
        
        # 훈련 상태 저장
        training_state = self.training
        self.eval()
        
        # 프롬프트 토큰화 - 간단한 문자열 사용
        
        prompt_text = self.tokenizer.apply_chat_template(
            self.caption_prompt_template,
            tokenize=False,
            return_tensors=None
        )

        prompt_tokens = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True
        ).to(v_emb.device)

        # 입력 ID가 Long 타입인지 확인
        prompt_tokens.input_ids = prompt_tokens.input_ids.long()
        # 프롬프트 임베딩
        prompt_embeds = self.llm.get_input_embeddings()(prompt_tokens.input_ids)
        
        # 각 샘플에 대해 개별 처리
        for b in range(batch_size):
            # 개별 샘플의 특징 추출
            sample_features = v_emb[b].unsqueeze(0)  # (1, seq_len, dim)
            
            # 비주얼 특징의 평균값을 컨텍스트로 사용
            visual_context = sample_features.mean(dim=1, keepdim=True)  # (1, 1, dim)
            
            # 비주얼 컨텍스트와 프롬프트 결합
            combined_embeds = torch.cat([visual_context, prompt_embeds.repeat(1, 1, 1)], dim=1)
            
            # 캡션 생성
            with torch.no_grad():
                attention_mask = torch.ones(
                    combined_embeds.shape[:2], 
                    dtype=torch.long, 
                    device=combined_embeds.device
                )
                
                outputs = self.llm.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,  # 명시적 어텐션 마스크 추가
                    max_new_tokens=30,
                    min_new_tokens=10,  # 최소 10개 토큰 생성
                    num_beams=3,        # 빔 수 증가
                    early_stopping=True,
                    do_sample=True,     # 샘플링 활성화
                    temperature=0.4,    # 약간 낮은 온도
                    top_p=0.9,
                    return_dict_in_generate=True,
                )
                
                outputs_list.append(outputs)
            
            # 프롬프트 길이 계산
            prompt_len = prompt_tokens.input_ids.shape[1]
            
            # 생성된 텍스트에서 프롬프트 이후 부분만 사용
            if outputs.sequences.shape[1] > prompt_len + 1:  # +1은 시작 토큰
                caption_only_ids = outputs.sequences[:, prompt_len + 1:]
                caption_text = self.tokenizer.decode(caption_only_ids[0], skip_special_tokens=True)
                # print(f"샘플 {b+1} 생성된 캡션: {caption_text}")
            else:
                caption_text = "A visual scene with various elements."
                caption_only_ids = self.tokenizer(
                    caption_text, return_tensors="pt"
                ).input_ids.to(v_emb.device)
            
            # 캡션 토큰을 임베딩으로 변환
            caption_embeds = self.llm.get_input_embeddings()(caption_only_ids)
            caption_embeds_list.append(caption_embeds.squeeze(0))
        
        # 원래 훈련 상태로 복원
        self.train(training_state)
        
        return caption_embeds_list, outputs_list

    def _interleave_features_and_captions(self, v_emb, caption_embeds_list):
        """비주얼 특징과 캡션을 인터리빙하는 메서드"""
        device = v_emb.device
        dtype = v_emb.dtype
        
        # v_emb 차원 확인 및 처리
        if v_emb.dim() == 2:
            # newline_inserter가 반환한 2D 텐서인 경우 (flattened)
            num_features, dim = v_emb.shape
            
            # 2D -> 3D 변환 (batch 차원 추가)
            v_emb = v_emb.unsqueeze(0)  # [1, num_features, dim]
            batch_size = 1
            v_seq_len = num_features
        else:
            # 이미 3D 텐서인 경우
            batch_size, v_seq_len, dim = v_emb.shape
        
        # 각 샘플에 대해 특징과 캡션을 인터리빙
        interleaved_features = []
        
        for b in range(batch_size):
            sample_features = v_emb[b]  # [seq_len, dim]
            sample_caption = caption_embeds_list[b]
            
            # 캡션 차원 확인 및 조정
            if sample_caption.dim() == 3:
                sample_caption = sample_caption.squeeze(0)  # 배치 차원 제거
            
            # 비주얼 특징을 4개 청크로 분할
            chunk_size = v_seq_len // 4
            interleaved = []
            
            for i in range(0, v_seq_len, chunk_size):
                end = min(i + chunk_size, v_seq_len)
                # 현재 청크의 비주얼 특징 추가
                interleaved.append(sample_features[i:end])
                
                # 마지막 청크가 아니면 현재 위치에 캡션 추가
                if end < v_seq_len:
                    interleaved.append(sample_caption)
            
            # 인터리빙된 특징을 하나로 결합
            interleaved_sample = torch.cat(interleaved, dim=0)
            interleaved_features.append(interleaved_sample)
        
        # 배치 내 모든 샘플이 동일한 길이를 갖도록 패딩
        max_length = max(feat.shape[0] for feat in interleaved_features)
        padded_features = []
        
        for feat in interleaved_features:
            current_length = feat.shape[0]
            if current_length < max_length:
                # 패딩 추가
                padding = torch.zeros(max_length - current_length, dim, 
                                    device=device, dtype=dtype)
                padded = torch.cat([feat, padding], dim=0)
            else:
                padded = feat
            padded_features.append(padded)
        
        # 배치 차원으로 스택
        result = torch.stack(padded_features)
        return result

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
                
            # 5. 각 비디오 샘플에 대한 캡션 생성
            caption_embeds_list, outputs_list = self._generate_captions_for_features(v_emb)
                
            # 6. 차원 문제 방지를 위해 뉴라인 토큰 삽입 없이 인터리빙만 수행
            v_emb = self._interleave_features_and_captions(v_emb, caption_embeds_list)
            
            v_embs[i] = v_emb
        
        # 7. 이미지 토큰 대체
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