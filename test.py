import torch
from src.models.config import VisionLanguageConfig
from transformers import AutoTokenizer
import os

# 상수 직접 정의
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# CaptioningVLM 클래스 불러오기
from src.models.captionvlm import CaptioningVLM

def test_captioning_vlm():
    print("CaptioningVLM 모델 테스트를 시작합니다...")
    
    # 1. 설정 생성
    cfg = VisionLanguageConfig(
        vision_model_name="facebook/dino-vitb16",
        language_model_name="Qwen/Qwen3-0.6B",
        projector_type="mlp2x_gelu",
        use_resampler=False,
        mm_spatial_pool_mode="average",
        freeze_vision=True,
        freeze_llm=True,
    )
    
    # 2. 모델 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    print(f"디바이스: {device} 사용")
    
    model = CaptioningVLM(cfg, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
    model = model.to(device)
    model.eval()  # 평가 모드로 설정
    
    # 3. 더미 입력 데이터 생성
    dummy_frames = torch.randn(6, 3, 224, 224, device=device, dtype=torch.float32)
    
    # 6. 내부 캡셔닝 과정 먼저 테스트 (다른 테스트 오류 때문에)
    print("\n[캡셔닝 과정 테스트]")
    with torch.no_grad():
        try:
            # 비전 임베딩 직접 추출
            v_embs = model._get_vision_embeds(dummy_frames)
            print(f"비전 임베딩 shape: {v_embs.shape}")
            
            # 풀링 적용
            pooled_embs = model._get_2dPool(v_embs)
            print(f"풀링 후 shape: {pooled_embs.shape}")
            
            # 캡션 생성 직접 테스트
            caption_embeds, outputs_list = model._generate_captions_for_features(pooled_embs)
            
            print(f"생성된 캡션 수: {len(caption_embeds)}")
            for i, emb in enumerate(caption_embeds):
                print(f"샘플 {i+1} 캡션 임베딩 shape: {emb.shape}")
            
            # 생성된 캡션 확인
            for i, output in enumerate(outputs_list):
                caption_text = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                print(f"샘플 {i+1} 전체 생성 텍스트: {caption_text}")
            
            # 인터리빙 테스트
            interleaved = model._interleave_features_and_captions(pooled_embs, caption_embeds)
            print(f"인터리빙 후 shape: {interleaved.shape}")
            
        except Exception as e:
            print(f"캡셔닝 과정 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_captioning_vlm()