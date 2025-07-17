import os
import argparse
import torch
import json
import gc
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoProcessor,
)
from copy import deepcopy
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback, AutoTokenizer
import yaml
from omegaconf import OmegaConf
from types import SimpleNamespace

from src.dataset import VLMDataset
from src.models.config import VisionLanguageConfig
from src.models.build import CustomVLMModel
from src.models.captionvlm import CaptioningVLM
from src.models.constant import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# ---------------------------------------------------------------------------- #
# GPU 메모리 관리 유틸리티
# ---------------------------------------------------------------------------- #
def print_gpu_memory_info(prefix=""):
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU 메모리: 할당={allocated:.2f}GB, 캐시={cached:.2f}GB")
    
def clear_gpu_cache():
    """GPU 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def estimate_memory_requirements(batch_size, sequence_length, hidden_size):
    """메모리 요구량 추정 (대략적)"""
    # 간단한 추정식 (실제로는 더 복잡함)
    estimated_gb = (batch_size * sequence_length * hidden_size * 4) / (1024**3)  # float32 기준
    return estimated_gb

def safe_json_serialize(obj):
    """JSON 직렬화가 안전한 객체로 변환"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif hasattr(obj, 'name'):  # Enum 타입
        return obj.name
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)

# ---------------------------------------------------------------------------- #
# 0. Prompt Formatter
# ---------------------------------------------------------------------------- #
def create_input_with_template(instruction, tokenizer, image_placeholder=DEFAULT_IMAGE_TOKEN):
    """
    주어진 instruction과 템플릿을 결합한 후 토큰화된 결과를 반환하는 함수.

    Args:
    - instruction (str): 사용자가 입력한 지시문.
    - image_placeholder (str): <image>로 대체될 이미지 토큰. 기본값은 "<image>".

    Returns:
    - dict: 토크나이저에 의해 인코딩된 입력.
    """
    # 템플릿을 정의 (여기서는 단순 예시)
    template = f"Here is an image prompt: {image_placeholder} Now answer the following question: {instruction}"

    # 템플릿과 instruction을 결합하여 토큰화
    inputs = tokenizer(template, return_tensors="pt")

    return inputs

class LoRACheckpointCallback(TrainerCallback):
    """LoRA 가중치와 관련 구성요소를 포함한 완전한 체크포인트 저장 콜백"""
    
    def __init__(self, vision_processor, tokenizer, model_config, lora_config):
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.lora_config = lora_config

    def on_save(self, args, state, control, model=None, **kwargs):
        """체크포인트 저장 시 호출되는 메서드"""
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        print(f"\n=== LoRA 체크포인트 저장 시작: {ckpt_dir} ===")
        print_gpu_memory_info("저장 전")
        
        # 1. LoRA 어댑터 저장
        print("1. LoRA 어댑터 가중치 저장...")
        model.save_pretrained(ckpt_dir)
        
        # 2. 토크나이저 저장 (특수 토큰 포함)
        print("2. 토크나이저 저장 (특수 토큰 포함)...")
        self.tokenizer.save_pretrained(ckpt_dir)
        
        # 3. 비전 프로세서 저장
        print("3. 비전 프로세서 저장...")
        self.vision_processor.save_pretrained(ckpt_dir)
        
        # 4. 모델 설정 저장
        print("4. VLM 모델 설정 저장...")
        model_config_path = os.path.join(ckpt_dir, "vlm_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(self.model_config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 5. LoRA 설정 저장 (JSON 직렬화 가능한 형태로)
        print("5. LoRA 설정 저장...")
        try:
            lora_config_dict = safe_json_serialize(self.lora_config.__dict__)
            lora_config_path = os.path.join(ckpt_dir, "lora_config.json")
            with open(lora_config_path, 'w') as f:
                json.dump(lora_config_dict, f, indent=2, ensure_ascii=False)
            print(f"   ✓ LoRA 설정 저장 완료: {len(lora_config_dict)} 개 항목")
        except Exception as e:
            print(f"   ❌ LoRA 설정 저장 실패: {e}")
            # PEFT의 기본 설정 파일이 이미 저장되었으므로 계속 진행
        
        # 6. 체크포인트 메타데이터 저장
        print("6. 체크포인트 메타데이터 저장...")
        try:
            metadata = {
                "step": state.global_step,
                "epoch": state.epoch if hasattr(state, 'epoch') else 0.0,
                "model_type": "CaptioningVLM_with_LoRA",
                "base_model": self.model_config.language_model_name,
                "vision_model": self.model_config.vision_model_name,
                "lora_r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": safe_json_serialize(self.lora_config.target_modules),
                "save_timestamp": f"step_{state.global_step}_epoch_{state.epoch:.2f}" if hasattr(state, 'epoch') else f"step_{state.global_step}",
                "training_loss": getattr(state, 'log_history', [])[-1].get('train_loss', 0.0) if hasattr(state, 'log_history') and state.log_history else 0.0
            }
            
            # 전체 메타데이터를 안전하게 직렬화
            safe_metadata = safe_json_serialize(metadata)
            
            metadata_path = os.path.join(ckpt_dir, "checkpoint_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
            print(f"   ✓ 메타데이터 저장 완료: step {state.global_step}")
        except Exception as e:
            print(f"   ❌ 메타데이터 저장 실패: {e}")
            # 기본 메타데이터라도 저장 시도
            basic_metadata = {
                "step": int(state.global_step),
                "epoch": float(state.epoch) if hasattr(state, 'epoch') else 0.0,
                "model_type": "CaptioningVLM_with_LoRA"
            }
            try:
                metadata_path = os.path.join(ckpt_dir, "checkpoint_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(basic_metadata, f, indent=2, ensure_ascii=False)
                print("   ✓ 기본 메타데이터 저장 완료")
            except Exception as e2:
                print(f"   ❌ 기본 메타데이터 저장도 실패: {e2}")
        
        # 7. 로드 스크립트 생성
        print("7. 로드 도우미 스크립트 생성...")
        self._create_load_script(ckpt_dir)
        
        # 8. 메모리 정리
        clear_gpu_cache()
        print_gpu_memory_info("저장 후")
        
        print(f"=== LoRA 체크포인트 저장 완료: {ckpt_dir} ===\n")
    
    def _create_load_script(self, ckpt_dir):
        """체크포인트 로드를 위한 도우미 스크립트 생성"""
        load_script = f'''
# LoRA 체크포인트 로드 예시 스크립트
import torch
import os
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
import json
from src.models.config import VisionLanguageConfig
from src.models.captionvlm import CaptioningVLM

def safe_load_config(config_dict):
    """안전하게 LoRA 설정을 로드하는 함수"""
    # task_type 처리
    if isinstance(config_dict.get('task_type'), str):
        try:
            config_dict['task_type'] = getattr(TaskType, config_dict['task_type'])
        except AttributeError:
            config_dict['task_type'] = TaskType.CAUSAL_LM
    
    # target_modules 처리
    target_modules = config_dict.get('target_modules', [])
    if isinstance(target_modules, str):
        try:
            config_dict['target_modules'] = eval(target_modules)
        except:
            config_dict['target_modules'] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return config_dict

def load_lora_checkpoint(checkpoint_path="{ckpt_dir}"):
    """LoRA 체크포인트를 로드하는 함수"""
    
    try:
        # 1. 설정 파일 로드
        with open(os.path.join(checkpoint_path, "vlm_config.json"), 'r') as f:
            model_config_dict = json.load(f)
        
        with open(os.path.join(checkpoint_path, "lora_config.json"), 'r') as f:
            lora_config_dict = json.load(f)
        
        # 2. 토크나이저 및 프로세서 로드
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        vision_processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        # 3. VLM 설정 재구성
        model_config = VisionLanguageConfig(**model_config_dict)
        
        # 4. LoRA 설정 재구성 (안전한 타입 변환)
        safe_lora_config = safe_load_config(lora_config_dict)
        lora_config = LoraConfig(**safe_lora_config)
        
        # 5. 베이스 모델 로드
        base_model = CaptioningVLM(config=model_config, tokenizer=tokenizer)
        
        # 6. LoRA 어댑터 적용
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        return model, tokenizer, vision_processor
        
    except Exception as e:
        print(f"체크포인트 로드 실패: {{e}}")
        print("PEFT adapter_config.json을 사용한 기본 로드를 시도합니다...")
        
        # 기본 로드 방식
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        vision_processor = AutoProcessor.from_pretrained(checkpoint_path)
        
        # adapter_config.json에서 base_model_name_or_path 가져오기
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-0.6B')
        else:
            base_model_name = 'Qwen/Qwen3-0.6B'
        
        # 기본 설정으로 VLM 생성
        model_config = VisionLanguageConfig(
            language_model_name=base_model_name,
            vision_model_name="google/siglip-so400m-patch14-384"
        )
        
        base_model = CaptioningVLM(config=model_config, tokenizer=tokenizer)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        return model, tokenizer, vision_processor

def load_for_inference(checkpoint_path="{ckpt_dir}"):
    """추론용으로 모델을 로드하는 간단한 함수"""
    model, tokenizer, processor = load_lora_checkpoint(checkpoint_path)
    model.eval()
    return model, tokenizer, processor

# 사용 예시:
# model, tokenizer, processor = load_lora_checkpoint()
# model.eval()

# 또는 추론용:
# model, tokenizer, processor = load_for_inference()
'''
        
        script_path = os.path.join(ckpt_dir, "load_checkpoint.py")
        with open(script_path, 'w') as f:
            f.write(load_script)

# ---------------------------------------------------------------------------- #
# 2. Collator
# ---------------------------------------------------------------------------- #
class MultimodalCollator:
    def __init__(self, tokenizer=None, pad_token_id=0):
        self.tokenizer = tokenizer
        # 토크나이저가 제공된 경우 해당 pad_token_id 사용
        self.pad_token_id = tokenizer.pad_token_id if tokenizer is not None else pad_token_id
    
    def __call__(self, features):
        # 1. 비디오 프레임 처리 (pixel_values)
        # 프레임 차원 구조 확인 및 배치 구성
        frames, c, h, w = features[0]['pixel_values'].shape
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        
        # 2. 텍스트 데이터 패딩 및 처리
        # 최대 길이 계산
        max_length = max(f['input_ids'].size(0) for f in features)
        
        # 패딩된 배치 텐서 초기화
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_length), 
                               self.pad_token_id, 
                               dtype=features[0]['input_ids'].dtype,
                               device=features[0]['input_ids'].device)
        attention_mask = torch.zeros((batch_size, max_length), 
                                    dtype=features[0]['attention_mask'].dtype,
                                    device=features[0]['attention_mask'].device)
        
        # 각 샘플에 대해 패딩 적용
        for i, f in enumerate(features):
            input_len = f['input_ids'].size(0)
            input_ids[i, :input_len] = f['input_ids']
            attention_mask[i, :input_len] = f['attention_mask']
        
        # 기본 배치 구성
        batch = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # 3. 라벨 처리 (존재하는 경우)
        if 'labels' in features[0]:
            # 기본적으로 -100으로 초기화된 라벨 텐서 생성
            labels = torch.full((batch_size, max_length), 
                               -100,  # IGNORE_INDEX
                               dtype=features[0]['labels'].dtype,
                               device=features[0]['labels'].device)
            
            # 각 샘플의 라벨 복사
            for i, f in enumerate(features):
                label_len = f['labels'].size(0)
                labels[i, :label_len] = f['labels']
                
            # 주의: 라벨은 이미 -100으로 마스킹되어 있으므로 
            # 추가 패딩 마스킹은 필요하지 않음
            batch['labels'] = labels
        
        # 4. 비텐서 데이터 처리 (경로 등)
        if 'video_path' in features[0]:
            batch['video_path'] = [f['video_path'] for f in features]
        
        return batch

# ---------------------------------------------------------------------------- #
# 3. Argument Parser
# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM with parameters from YAML")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the YAML config")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="Path to LoRA checkpoint to resume from")
    return parser.parse_args()

def load_config(config_path: str):
    """
    Load YAML into an OmegaConf DictConfig so we can keep dot‑access,
    then resolve any {model_name} placeholders.
    """
    cfg = OmegaConf.load(config_path)

    # Resolve {model_name} templates
    model_name = cfg.model.name
    cfg.training.output_dir = cfg.training.output_dir.format(model_name=model_name)
    cfg.training.run_name   = cfg.training.run_name.format(model_name=model_name)
    cfg.training.logging_dir = cfg.training.logging_dir.format(model_name=model_name)
    # Backward compatibility: allow config with only "data:" section
    if "dataset" not in cfg and "data" in cfg:
        cfg.dataset = cfg.data  # backward compatibility alias
    return cfg

def load_lora_checkpoint(checkpoint_path, model_config, lora_config):
    """LoRA 체크포인트에서 모델을 로드하는 함수"""
    print(f"LoRA 체크포인트에서 모델 로드: {checkpoint_path}")
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # 베이스 모델 생성
        base_model = CaptioningVLM(config=model_config, tokenizer=tokenizer)
        
        # LoRA 어댑터 적용
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        print("LoRA 체크포인트 로드 완료")
        return model, tokenizer
        
    except Exception as e:
        print(f"체크포인트 로드 중 오류 발생: {e}")
        print("기본 설정으로 새 모델을 생성합니다...")
        
        # 토크나이저 로드 시도
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            # 체크포인트에서 토크나이저 로드 실패 시 기본 모델에서 로드
            tokenizer = AutoTokenizer.from_pretrained(model_config.language_model_name)
            
            # 토크나이저 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            # 특수 토큰 추가
            special_tokens = {"additional_special_tokens": [
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_PATCH_TOKEN, 
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
            ]}
            tokenizer.add_special_tokens(special_tokens)
        
        # 새 모델 생성
        base_model = CaptioningVLM(config=model_config, tokenizer=tokenizer)
        from peft import get_peft_model
        model = get_peft_model(base_model, lora_config)
        
        return model, tokenizer

# ---------------------------------------------------------------------------- #
# 4. Main training flow
# ---------------------------------------------------------------------------- #
def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    print("=== 훈련 설정 정보 ===")
    print(f"배치 크기: {cfg.training.batch_size.train}")
    print(f"Gradient Accumulation Steps: {cfg.training.gradient_accumulation_steps}")
    print(f"실제 배치 크기: {cfg.training.batch_size.train * cfg.training.gradient_accumulation_steps}")
    print_gpu_memory_info("초기")
    
    # Config 설정
    model_config = VisionLanguageConfig(
        vision_model_name=cfg.model.vision_model_name,
        language_model_name=cfg.model.llm_model_name,
        projector_type=cfg.model.projector_type,
        use_resampler=cfg.model.use_resampler,
        mm_spatial_pool_mode=cfg.model.mm_spatial_pool_mode,
        mm_newline_position=getattr(cfg.model, "mm_newline_position", "grid"),
        freeze_vision=cfg.model.freeze_vision,
        freeze_llm=cfg.model.freeze_llm,
    )

    vision_processor = AutoProcessor.from_pretrained(cfg.model.vision_model_name)
    print_gpu_memory_info("프로세서 로드 후")
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = 128,
        lora_alpha      = 64,
        lora_dropout    = 0.1,
        base_model_name_or_path = cfg.model.llm_model_name,
        target_modules  = [
            "q_proj", "k_proj", "v_proj",          # attention
            "o_proj",
            "gate_proj", "up_proj", "down_proj"    # MLP
        ],
        bias            = "none"
    )
    
    # 체크포인트에서 재개하는 경우와 새로 시작하는 경우 구분
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        # 체크포인트에서 로드
        model, language_processor = load_lora_checkpoint(
            args.resume_from_checkpoint, model_config, lora_config
        )
        print(f"체크포인트에서 훈련 재개: {args.resume_from_checkpoint}")
    else:
        # 새로운 모델 생성
        language_processor = AutoTokenizer.from_pretrained(cfg.model.llm_model_name)

        # 토크나이저 설정
        if language_processor.pad_token is None:
            language_processor.pad_token = language_processor.eos_token
        language_processor.padding_side = "right"

        # 특수 토큰 추가
        special_tokens = {"additional_special_tokens": [
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IMAGE_PATCH_TOKEN, 
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        ]}
        language_processor.add_special_tokens(special_tokens)

        print_gpu_memory_info("토크나이저 설정 후")

        # 모델 초기화
        model = CaptioningVLM(
            config=model_config, 
            tokenizer=language_processor
        )
        print_gpu_memory_info("모델 초기화 후")

        # LoRA 적용
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("새로운 LoRA 모델 생성 완료")
        print_gpu_memory_info("LoRA 적용 후")

    # Dataset 설정
    ds_cfg = cfg.dataset
    train_ds = VLMDataset(
        data_path           = ds_cfg.data_path,
        data_files          = getattr(ds_cfg, "data_files", getattr(ds_cfg, "train_file", None)),
        image_placeholder   = DEFAULT_IMAGE_TOKEN,
        max_frames_num      = getattr(ds_cfg, "max_frames_num", 64),
        fps                 = getattr(ds_cfg, "fps", 1),
        img_processor       = vision_processor,
        tokenizer           = language_processor,
        force_sample        = getattr(ds_cfg, "force_sample", False),
    )
    print(f"데이터셋 크기: {len(train_ds)}")

    # GPU로 이동 및 메모리 최적화
    if torch.cuda.is_available():
        model.cuda()
        # 메모리 절약을 위해 gradient checkpointing 활성화
        model.gradient_checkpointing_enable()
        print(f"모델을 GPU로 이동, Gradient Checkpointing 활성화")
        print_gpu_memory_info("GPU 이동 후")
        
        # 메모리 부족 시 권장사항 출력
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        if allocated_gb > 8:  # 8GB 이상 사용 시 경고
            print("\n⚠️  GPU 메모리 사용량이 높습니다. 다음 설정을 고려해보세요:")
            print(f"   - 배치 크기를 {cfg.training.batch_size.train // 2}로 줄이기")
            print(f"   - Gradient accumulation steps를 {cfg.training.gradient_accumulation_steps * 2}로 늘리기")
            print("   - max_frames_num을 32로 줄이기")
            print("   - DeepSpeed ZeRO 사용하기")
    else:
        print("CUDA를 사용할 수 없습니다. CPU에서 훈련을 진행합니다.")
    
    # 훈련 설정 - 메모리 효율성 개선
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        logging_dir=cfg.training.logging_dir,
        deepspeed=cfg.deepspeed.config if "deepspeed" in cfg and cfg.deepspeed.enabled else None,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.batch_size.train,
        
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=True,  # 메모리 절약을 위해 활성화
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        
        # 메모리 효율성 개선 옵션들
        fp16=True,  # Mixed precision training 활성화
        dataloader_pin_memory=False,  # 메모리 절약
        remove_unused_columns=False,  # 필요한 컬럼 보존
        
        # 체크포인트 및 로깅 설정
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=False,

        report_to=cfg.training.report_to,
        logging_steps=cfg.training.logging_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        
        # 레이블 관련 설정
        label_names=["labels"],  # PeftModel 경고 해결
    )
    
    # 트레이너 설정
    data_collator = MultimodalCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=language_processor,  # tokenizer 대신 processing_class 사용
    )
    
    # LoRA 체크포인트 콜백 추가
    lora_callback = LoRACheckpointCallback(
        vision_processor=vision_processor,
        tokenizer=language_processor,
        model_config=model_config,
        lora_config=lora_config
    )
    trainer.add_callback(lora_callback)
    
    # 훈련 시작
    try:
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU 메모리 부족 오류 발생: {e}")
        print("\n🔧 메모리 절약을 위한 권장 설정:")
        print("1. config/train.yaml에서 다음 값들을 줄여보세요:")
        print(f"   - batch_size.train: {cfg.training.batch_size.train} → {max(1, cfg.training.batch_size.train // 2)}")
        print(f"   - max_frames_num: {getattr(cfg.dataset, 'max_frames_num', 64)} → 32")
        print("2. gradient_accumulation_steps를 늘려서 effective batch size 유지")
        print("3. DeepSpeed ZeRO Stage 2 활성화")
        print("4. 더 작은 모델 사용 고려")
        raise
    
    # 최종 병합 모델 저장
    print("\n=== 최종 병합 모델 저장 시작 ===")
    print_gpu_memory_info("병합 전")
    
    try:
        merged_model = model.merge_and_unload()
        print_gpu_memory_info("병합 후")
        
        save_dir = os.path.join(training_args.output_dir, "merged_final")
        os.makedirs(save_dir, exist_ok=True)
        
        merged_model.save_pretrained(save_dir, safe_serialization=True)
        language_processor.save_pretrained(save_dir)
        vision_processor.save_pretrained(save_dir)
        
        # 최종 모델 설정도 저장
        final_config_path = os.path.join(save_dir, "vlm_config.json")
        with open(final_config_path, 'w') as f:
            json.dump(model_config.to_dict(), f, indent=2, ensure_ascii=False)
        
        clear_gpu_cache()
        print(f"=== 최종 병합 모델 저장 완료: {save_dir} ===")
        print_gpu_memory_info("최종")
        
    except Exception as e:
        print(f"❌ 최종 모델 저장 중 오류 발생: {e}")
        print("LoRA 어댑터만 저장된 체크포인트를 사용하세요.")
        raise

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()