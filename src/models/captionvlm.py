from typing import Optional
import torch
from src.models.build import CustomVLMModel
from typing import Union, List, Tuple

# 필요한 상수들을 직접 정의 (import 문제 해결)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class CaptioningVLM(CustomVLMModel):
    """
    비주얼 콘텐츠에 대한 자동 캡션 생성 기능을 갖춘 비전-언어 모델(Vision-Language Model).
    이 모델은 표준 VLM의 기능을 확장하여 비주얼 특징(이미지 또는 비디오 프레임)에 대한 
    자동 캡션을 생성하고, 이를 원본 비주얼 특징과 함께 사용하여 향상된 멀티모달 이해를 제공합니다.
    주요 기능:
    1. 비주얼 특징에 대한 자동 캡션 생성
    2. 비주얼 특징과 생성된 캡션의 인터리빙(번갈아 배치)
    3. 멀티모달 입력 전처리 및 토큰 대체
    주요 메소드:
    - _generate_captions_for_features: 비주얼 특징으로부터 텍스트 캡션을 생성
        입력: v_emb (torch.FloatTensor) - [batch_size, seq_len, dim] 형태의 비주얼 특징
        출력: (list[torch.FloatTensor], list) - 캡션 임베딩 리스트와 생성 결과 리스트
    - _interleave_features_and_captions: 비주얼 특징과 캡션을 번갈아 배치
        입력: v_emb (torch.FloatTensor) - [batch_size, seq_len, dim] 또는 [num_features, dim] 형태의 비주얼 특징
                    caption_embeds_list (list[torch.FloatTensor]) - 각 샘플의 캡션 임베딩
        출력: torch.FloatTensor - [batch_size, max_seq_len, dim] 형태의 인터리빙된 특징
    - _prepare_multimodal_inputs: 멀티모달 입력을 준비하고 이미지 토큰을 특징으로 대체
        입력: pixel_values (torch.FloatTensor) - [batch_size, channels, height, width] 또는 
            [batch_size, frames, channels, height, width] 형태의 이미지/비디오
            input_ids (torch.LongTensor) - [batch_size, seq_len] 형태의 입력 토큰 ID
            attention_mask (torch.LongTensor, 선택적) - 어텐션 마스크
            labels (torch.LongTensor, 선택적) - 레이블
        출력: (torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor) - 
                    (입력 임베딩, 어텐션 마스크, 레이블, 포지션 ID)
    기술적 특징:
    - 비주얼 특징을 4개 청크로 분할하여 처리
    - 캡션 생성 시 빔 서치와 샘플링을 결합하여 품질 향상
    - 자동 패딩을 통한 일괄 처리 지원
    - 뉴라인 토큰을 사용한 구조적 특징 구성
    """
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
    def _generate_captions_for_features(self, v_emb: torch.FloatTensor):
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
        labels: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        # 비디오인 경우 첫 4프레임만 사용
        if pixel_values.dim() == 5:
            pixel_values = pixel_values[:, :4, ...]
        
        # 1. 토큰 전처리
        processed_input_ids = self.preprocess_image_tokens(input_ids)
        processed_labels = processed_input_ids.clone() if labels is None else self.preprocess_image_tokens(labels)
        
        # 2. 비전 인코딩
        B = processed_input_ids.size(0)
        v_embs = self._get_vision_embeds(pixel_values)
        v_embs = list(torch.split(v_embs, B, dim=0))
        
        for i, v_emb in enumerate(v_embs):
            # 3. 비전 임베딩 풀링
            if self.config.mm_spatial_pool_mode != "none":
                v_emb = self._get_2dPool(v_emb, stride=2)
            
            # 4. 비전 특징을 4개 청크로 분할
            batch_size, seq_len, dim = v_emb.shape
            processed_chunks = []
            
            for b in range(batch_size):
                sample_features = v_emb[b]  # (seq_len, dim)
                chunk_size = seq_len // 4
                
                # 4개 청크로 분할 및 각 청크에 뉴라인 토큰 삽입
                chunks_with_newline = []
                for c in range(4):
                    start_idx = c * chunk_size
                    end_idx = (c + 1) * chunk_size if c < 3 else seq_len
                    chunk = sample_features[start_idx:end_idx]
                    
                    # 뉴라인 토큰 삽입 (NewlineTokenInserter 적용)
                    chunk_with_newline = self.newline_inserter(chunk, self.image_newline)
                    chunks_with_newline.append(chunk_with_newline)
                
                # 5. 캡션 생성
                caps_list, _ = self._generate_captions_for_features(v_emb[b:b+1])
                caption_embed = caps_list[0]
                
                # 6. 특징 청크와 캡션 결합 (캡션은 맨 뒤에 추가)
                combined_features = torch.cat(chunks_with_newline + [caption_embed], dim=0)
                processed_chunks.append(combined_features)
            
            # 배치 내 길이 맞추기
            max_len = max(x.shape[0] for x in processed_chunks)
            padded_chunks = []
            for chunk in processed_chunks:
                if chunk.shape[0] < max_len:
                    padding = torch.zeros(max_len - chunk.shape[0], dim, device=chunk.device, dtype=chunk.dtype)
                    padded = torch.cat([chunk, padding], dim=0)
                else:
                    padded = chunk
                padded_chunks.append(padded)
            
            # 배치 차원으로 다시 결합
            v_embs[i] = torch.stack(padded_chunks, dim=0)
        
        # 7. 이미지 토큰 대체
        return self._replace_image_tokens_with_features(
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