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
        self.captioning_instruction = "<image> Generate a short descriptive caption for this visual content."
        self.caption_prompt_template = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": self.captioning_instruction},
        ]
    
    @torch.no_grad()
    def _generate_captions_for_features(self, v_emb: torch.FloatTensor):
        """
        하나의 v_emb에 대한 캡션생성 메서드.
        v_emb: 비주얼 특징 텐서 ((1, seq_len' + newlinetoken_num, dim')
        returns: 임베딩된 캡션 텐서 (caption_length, dim)
        """
        # 입력 텐서 차원 확인 및 조정
        if v_emb.dim() == 2:  # (seq_len, dim) 형태인 경우
            v_emb = v_emb.unsqueeze(0)  # (1, seq_len, dim)로 변환
        
        # 훈련 상태 저장
        training_state = self.training
        self.eval()
        
        # 프롬프트 토큰화
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
        # 수정: prompt_tokens 객체 자체가 아닌 input_ids 텐서를 전달
        processed_input_ids = self.preprocess_image_tokens(prompt_tokens.input_ids)
        
        # Dummy labels 준비 - 오류 수정: 필수 인자 추가, _replace_image_tokens_with_features에서 labels를 사용함
        # 캡션 생성 단계에서는 레이블이 필요하지 않으므로 무시 인덱스(-100)로 설정
        processed_labels = torch.full_like(processed_input_ids, IGNORE_INDEX)
    
        # 프롬프트 임베딩
        
        inp_emb, pad_lbl, pad_mask, pos_ids = self._replace_image_tokens_with_features(
            input_ids=processed_input_ids,
            labels=processed_labels,  # 필수 인자 labels 추가
            attention_mask=prompt_tokens.attention_mask,
            image_features=[v_emb],
            embed_tokens_fn=self.llm.get_input_embeddings(),
            image_token_index=IMAGE_TOKEN_INDEX,
            ignore_index=IGNORE_INDEX,
            max_length=self.config.language_config.max_position_embeddings,
            padding_side=self.tokenizer.padding_side,
        )
        
        # 캡션 생성
        with torch.no_grad():
            attention_mask = torch.ones(
                inp_emb.shape[:2], 
                dtype=torch.long, 
                device=inp_emb.device
            )
            
            outputs = self.llm.generate(
                inputs_embeds=inp_emb,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                max_new_tokens=30,
                min_new_tokens=10,
                num_beams=3,
                early_stopping=True,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                return_dict_in_generate=True,
            )
        
        # 프롬프트 길이 계산
        prompt_len = processed_input_ids.shape[1] 
        
        # 생성된 텍스트에서 프롬프트 이후 부분만 사용
        if outputs.sequences.shape[1] > prompt_len + 1:
            caption_only_ids = outputs.sequences[:, prompt_len + 1:]
            caption_text = self.tokenizer.decode(caption_only_ids[0], skip_special_tokens=True)
        else:
            caption_text = ""
            caption_only_ids = self.tokenizer(
                caption_text, return_tensors="pt"
            ).input_ids.to(v_emb.device).long()  # 명시적으로 long 타입으로 변환
        
        print(f"생성된 캡션: {caption_text}")
        
        # 캡션 토큰을 임베딩으로 변환
        caption_embeds = self.llm.get_input_embeddings()(caption_only_ids.long())
        
        # 원래 훈련 상태로 복원
        self.train(training_state)
        
        # 배치 차원 제거하고 반환
        return caption_embeds.squeeze(0) # (caption_length, dim), outputs

    def _prepare_multimodal_inputs(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        # 1. 토큰 전처리
        processed_input_ids = self.preprocess_image_tokens(input_ids)
        processed_labels = processed_input_ids.clone() if labels is None else self.preprocess_image_tokens(labels)
        
        # 2. 비전 인코딩
        # 계산 효율성을 위해 배치와 샘플들을 배치차원으로 결합해 처리한다.
        B = processed_input_ids.size(0) 
        v_embs = self._get_vision_embeds(pixel_values) # [B*num_saples, seq_len, dim']
        
        v_embs = list(torch.split(v_embs, v_embs.size(0)//B, dim=0)) #[(num_samples, seq_len, dim'), ...], len(v_embs) = B
        # 여기서 부터는 배치 단위로 따로 처리된다.
        # [(num_samples, seq_len, dim'), ...], len(v_embs) = B
        
        for i, v_emb in enumerate(v_embs):
            
            # 3. 비전 임베딩 풀링
            if self.config.mm_spatial_pool_mode != "none":
                v_emb = self._get_2dPool(v_emb, stride=2) # v_emb: [num_samples, seq_len', dim']

            chunk_num = 4
            num_samples, seq_len, dim = v_emb.shape
            chunk_size = num_samples // chunk_num if num_samples >= chunk_num else 1
            
            # 4개 청크로 분할 및 각 청크에 뉴라인 토큰 삽입
            chunks_with_caption = []
            for j in range(chunk_num):
                start = j * chunk_size
                end = (j + 1) * chunk_size if j < chunk_num - 1 else num_samples
                # 청크가 비어있지 않은지 확인
                if start < num_samples:
                    chunk = v_emb[start:end]  # (chunk_size, seq_len', dim')
                    print(f"청크 {j+1}: {chunk.shape}")
                    # 뉴라인 토큰 삽입
                    chunk_with_newline = self.newline_inserter(chunk, self.image_newline)
                    
                    # 캡션 생성 (self를 사용)
                    caption = self._generate_captions_for_features(chunk_with_newline)
                    
                    # 청크와 캡션 결합
                    chunk_with_caption = torch.cat([chunk_with_newline, caption], dim=0)
                    chunks_with_caption.append(chunk_with_caption)
                    
            # 배치 차원으로 다시 결합

            # 모든 청크 결합
            if chunks_with_caption:
                v_embs[i] = torch.cat(chunks_with_caption, dim=0)

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
        
    