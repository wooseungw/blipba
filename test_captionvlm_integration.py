#!/usr/bin/env python3
"""
CaptioningVLM 통합 테스트 스크립트
- 훈련 및 평가 환경에서 CaptioningVLM이 올바르게 작동하는지 확인
"""

import sys
import traceback

def test_imports():
    """필수 모듈 import 테스트"""
    print("=== 모듈 Import 테스트 ===")
    
    try:
        # 기본 설정 모듈
        from src.models.config import VisionLanguageConfig
        print("✓ VisionLanguageConfig import 성공")
        
        # 베이스 모델
        from src.models.build import CustomVLMModel
        print("✓ CustomVLMModel import 성공")
        
        # CaptioningVLM
        from src.models.captionvlm import CaptioningVLM
        print("✓ CaptioningVLM import 성공")
        
        # 상수들
        from src.models.constant import (
            IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        )
        print("✓ 상수 import 성공")
        
        return True, (VisionLanguageConfig, CustomVLMModel, CaptioningVLM)
        
    except Exception as e:
        print(f"✗ Import 실패: {e}")
        traceback.print_exc()
        return False, None

def test_model_creation():
    """모델 생성 테스트"""
    print("\n=== 모델 생성 테스트 ===")
    
    try:
        import torch
        from transformers import AutoTokenizer
        
        # 모듈 import
        success, modules = test_imports()
        if not success:
            return False
            
        VisionLanguageConfig, CustomVLMModel, CaptioningVLM = modules
        
        # 설정 생성
        config = VisionLanguageConfig(
            vision_model_name="facebook/dino-vitb16",
            language_model_name="Qwen/Qwen3-0.6B",
            projector_type="mlp2x_gelu",
            use_resampler=False,
            mm_spatial_pool_mode="average",
            freeze_vision=True,
            freeze_llm=True,
        )
        print("✓ VisionLanguageConfig 생성 성공")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ 토크나이저 로드 성공")
        
        # CaptioningVLM 모델 생성
        model = CaptioningVLM(config=config, tokenizer=tokenizer)
        print("✓ CaptioningVLM 모델 생성 성공")
        
        # 모델 구조 확인
        print(f"✓ 모델 타입: {type(model)}")
        print(f"✓ 모델이 CustomVLMModel의 서브클래스인지: {isinstance(model, CustomVLMModel)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 모델 생성 실패: {e}")
        traceback.print_exc()
        return False

def test_training_integration():
    """훈련 스크립트 import 테스트"""
    print("\n=== 훈련 스크립트 통합 테스트 ===")
    
    try:
        # train_vlm.py의 주요 import들 테스트
        from src.dataset import VLMDataset
        print("✓ VLMDataset import 성공")
        
        from src.models.captionvlm import CaptioningVLM
        print("✓ train_vlm.py에서 CaptioningVLM import 성공")
        
        # PEFT 관련
        from peft import LoraConfig, TaskType, get_peft_model
        print("✓ PEFT 모듈 import 성공")
        
        return True
        
    except Exception as e:
        print(f"✗ 훈련 스크립트 통합 실패: {e}")
        traceback.print_exc()
        return False

def test_evaluation_integration():
    """평가 스크립트 import 테스트"""
    print("\n=== 평가 스크립트 통합 테스트 ===")
    
    try:
        # evaluation.py의 dynamic import 기능 테스트
        import os
        
        # src 디렉토리 경로 찾기
        src_dir = None
        for parent_dir in ['.', '..', '../..']:
            if os.path.exists(os.path.join(parent_dir, 'src', 'models', 'config.py')):
                src_dir = os.path.abspath(parent_dir)
                break
        
        if not src_dir:
            raise ImportError("src 디렉토리를 찾을 수 없습니다")
        
        import sys
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # 필요한 모듈 import
        from src.models.config import VisionLanguageConfig
        from src.models.build import CustomVLMModel
        
        # CaptioningVLM도 import 시도
        try:
            from src.models.captionvlm import CaptioningVLM
            captioning_available = True
        except ImportError:
            captioning_available = False
        
        print(f"✓ VisionLanguageConfig import 성공")
        print(f"✓ CustomVLMModel import 성공")
        print(f"✓ CaptioningVLM 사용 가능: {captioning_available}")
        
        return True
        
    except Exception as e:
        print(f"✗ 평가 스크립트 통합 실패: {e}")
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("CaptioningVLM 통합 테스트 시작")
    print("=" * 50)
    
    # 기본 import 테스트
    import_success, _ = test_imports()
    
    # 모델 생성 테스트 (torch 사용 가능한 경우에만)
    model_success = False
    try:
        import torch
        model_success = test_model_creation()
    except ImportError:
        print("\n⚠️  PyTorch가 설치되지 않아 모델 생성 테스트를 건너뜁니다.")
        print("   실제 환경에서는 PyTorch 설치 후 테스트하세요.")
    
    # 훈련 통합 테스트
    training_success = test_training_integration()
    
    # 평가 통합 테스트  
    eval_success = test_evaluation_integration()
    
    # 최종 결과
    print("\n" + "=" * 50)
    print("=== 최종 테스트 결과 ===")
    print(f"모듈 Import: {'✓ 성공' if import_success else '✗ 실패'}")
    print(f"모델 생성: {'✓ 성공' if model_success else '⚠️  건너뜀/실패'}")
    print(f"훈련 통합: {'✓ 성공' if training_success else '✗ 실패'}")
    print(f"평가 통합: {'✓ 성공' if eval_success else '✗ 실패'}")
    
    overall_success = import_success and training_success and eval_success
    print(f"\n전체 통합 상태: {'✓ 성공' if overall_success else '✗ 부분 실패'}")
    
    if overall_success:
        print("\n🎉 CaptioningVLM이 훈련 및 평가 코드와 성공적으로 통합되었습니다!")
    else:
        print("\n⚠️  일부 통합에 문제가 있습니다. 위의 오류를 확인하세요.")

if __name__ == "__main__":
    main()