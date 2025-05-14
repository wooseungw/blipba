import os
import torch
from transformers import AutoProcessor
from peft import PeftModel

from src.models.config import VisionLanguageConfig
from src.models.build import CustomVLMModel
from src.constant import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def test_saved_model(model_path):
    print(f"모델 로딩 중: {model_path}")
    
    # 1. 설정과 모델 구조 로드
    print("설정 파일 로드 중...")
    # 먼저 원본 모델 구성을 로드 (merged_model을 로드하는 경우)
    try:
        # 일반 모델 로드 시도 (merged_model인 경우)
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            # 기본 설정 로드
            base_config = VisionLanguageConfig.from_pretrained(model_path)
            
            # 모델 인스턴스 생성
            model = CustomVLMModel.from_pretrained(
                model_path,
                config=base_config,
                vision_dtype=torch.float16,
                llm_dtype=torch.float16
            )
            is_peft_model = False
            print("통합 모델(merged) 로드됨")
        else:
            # LoRA 체크포인트인 경우
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(f"모델 구성 파일을 찾을 수 없습니다: {model_path}")
            
            # adapter_config.json에서 원본 모델 경로 추출
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                # 원본 모델 이름을 찾을 수 없는 경우, 체크포인트 이전 폴더에서 원본 모델 정보 찾기
                parent_dir = os.path.dirname(model_path)
                parent_config = os.path.join(parent_dir, "config.json")
                if os.path.exists(parent_config):
                    with open(parent_config, 'r') as f:
                        config_data = json.load(f)
                    vision_model_name = config_data.get("vision_model_name", "facebook/dino-vitb16")
                    language_model_name = config_data.get("language_model_name", "Qwen/Qwen3-0.6B")
                else:
                    # 기본값 사용
                    vision_model_name = "facebook/dino-vitb16"
                    language_model_name = "Qwen/Qwen3-0.6B"
                    print(f"Warning: 원본 모델 정보를 찾을 수 없어 기본값 사용: {vision_model_name}, {language_model_name}")
            else:
                # 원본 모델 경로 사용하여 설정 로드
                language_model_name = base_model_name
                # 비전 모델은 일단 기본값 사용
                vision_model_name = "facebook/dino-vitb16"
            
            # 모델 설정 구성
            model_config = VisionLanguageConfig(
                vision_model_name=vision_model_name,
                language_model_name=language_model_name,
                projector_type="mlp2x_gelu",
                use_resampler=False,
                mm_spatial_pool_mode="average",
            )
            
            # 베이스 모델 로드 
            base_model = CustomVLMModel(model_config)
            
            # PEFT 모델 로드
            from peft import PeftConfig, PeftModel
            model = PeftModel.from_pretrained(base_model, model_path)
            is_peft_model = True
            print("LoRA 체크포인트 모델 로드됨")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise
    
    # 2. 토크나이저 로드
    print("토크나이저 로드 중...")
    try:
        # 모델 내장 토크나이저 사용
        tokenizer = model.tokenizer
        print(f"토크나이저 로드됨: 어휘 크기 = {len(tokenizer)}")
        print(f"특수 토큰: {tokenizer.all_special_tokens}")
    except Exception as e:
        print(f"토크나이저 로드 중 오류 발생: {e}")
        raise
        
    # 3. 비전 프로세서 로드
    print("비전 프로세서 로드 중...")
    try:
        vision_processor = AutoProcessor.from_pretrained(model_path)
        print("비전 프로세서 로드됨")
    except Exception as e:
        print(f"비전 프로세서 로드 중 오류: {e}, 대체 프로세서 로드 시도")
        try:
            # 대체 방법으로 비전 모델 이름에서 직접 로드
            if hasattr(model.config, "vision_model_name"):
                vision_processor = AutoProcessor.from_pretrained(model.config.vision_model_name)
                print(f"대체 프로세서 로드됨: {model.config.vision_model_name}")
            else:
                vision_processor = AutoProcessor.from_pretrained("facebook/dino-vitb16")
                print("기본 비전 프로세서 로드됨: facebook/dino-vitb16")
        except Exception as sub_e:
            print(f"대체 프로세서 로드 중 오류: {sub_e}")
            raise
    
    # 모델을 평가 모드로 전환
    model.eval()
    
    # GPU 사용 가능하면 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 4. 더미 이미지 생성
    print(f"테스트용 더미 이미지 생성 중...")
    dummy_img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    
    # 5. 테스트 프롬프트 생성
    print("테스트 프롬프트 생성 중...")
    prompt = f"{DEFAULT_IM_START_TOKEN} What can you see in this image? {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}"
    
    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    # 6. 추론 실행
    print("추론 실행 중...")
    with torch.no_grad():
        try:
            # 일반 forward 실행
            outputs = model(
                pixel_values=dummy_img, 
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask
            )
            print(f"Forward 패스 성공: 출력 로짓 크기 = {outputs.logits.shape}")
            
            # 생성 실행
            try:
                generated_ids = model.generate(
                    pixel_values=dummy_img,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                )
                
                # 결과 출력
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                print("\n생성 결과:")
                print(generated_text[0])
                
                print("\n테스트 완료! 모델이 정상적으로 작동합니다.")
            except Exception as gen_e:
                print(f"생성 중 오류 발생: {gen_e}")
                
        except Exception as fwd_e:
            print(f"Forward 패스 중 오류 발생: {fwd_e}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="저장된 VLM 모델 테스트")
    parser.add_argument("--model_path", type=str, required=True,
                       help="테스트할 모델 경로 (merged_final 또는 checkpoint 디렉토리)")
    args = parser.parse_args()
    
    test_saved_model(args.model_path)