import os
import argparse
import torch
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

from fvcore.nn import FlopCountAnalysis, parameter_count_table
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

class CopyProcessorCallback(TrainerCallback):
    def __init__(self, vision_processor, tokenizer, model):
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        """체크포인트 저장 시 호출되는 메서드"""
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        print(f"\n저장 중: {ckpt_dir}")
        print("1. 토크나이저 저장 (특수 토큰 포함)")
        self.tokenizer.save_pretrained(ckpt_dir)
        
        print("2. 비전 프로세서 저장")
        self.vision_processor.save_pretrained(ckpt_dir)
        
        print("3. LoRA 모델 저장")
        self.model.save_pretrained(ckpt_dir)
        
        print(f"체크포인트 저장 완료: {ckpt_dir}\n")
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
# ---------------------------------------------------------------------------- #
# 4. Main training flow
# ---------------------------------------------------------------------------- #
def main():
    args = parse_args()
    cfg = load_config(args.config)
    # Config & Processor
    # 토크나이저 설정
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
    language_processor = AutoTokenizer.from_pretrained(cfg.model.llm_model_name)

    # 토크나이저에 특수 토큰 추가
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

    # 올바른 파라미터 이름으로 모델 초기화
    model = CaptioningVLM(
        config=model_config, 
        tokenizer=language_processor
    )



    # Dataset -----------------------------------------------------------------
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

    # LoRA -------------------------------------------------------------------
    # Qwen / OPT‑style transformer blocks expose projection names below.
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj",          # attention
        "o_proj",
        "gate_proj", "up_proj", "down_proj"    # MLP
    ]

    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = 128,
        lora_alpha      = 64,
        lora_dropout    = 0.1,
        base_model_name_or_path = cfg.model.llm_model_name,
        target_modules  = lora_target_modules,
        bias            = "none"
    )
    # PEFT 모델 생성 및 초기화 후 바로 체크
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Move model to GPU explicitly (Trainer would do this as well, but we place it early for PEFT initialization safety)
    if torch.cuda.is_available():
        model.cuda()
    model.gradient_checkpointing_disable()   # 모델 내부의 checkpoint 모듈 비활성화
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        logging_dir=cfg.training.logging_dir,
        deepspeed=cfg.deepspeed.config if "deepspeed" in cfg and cfg.deepspeed.enabled else None,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.batch_size.train,
        
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_num_workers=cfg.training.dataloader_num_workers,

        # per_device_eval_batch_size=cfg.training.batch_size.eval,
        # eval_strategy=cfg.training.eval_strategy,
        # eval_steps=cfg.training.eval_steps,

        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=False,   # we only keep the last N checkpoints

        report_to=cfg.training.report_to,
        logging_steps=cfg.training.logging_steps,
        max_grad_norm=cfg.training.max_grad_norm,
    )
    
    data_collator = MultimodalCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
    )
    # 콜백 추가 - 인자 이름에 주의!
    trainer.add_callback(CopyProcessorCallback(
        vision_processor=vision_processor,
        tokenizer=language_processor,  # 여기서는 'tokenizer' 매개변수 사용
        model=model
    ))

    dummy_inputs = {
    "input_ids": torch.randint(0, model.config.llm_vocab_size, (1, 256)).to(model.device),
    "attention_mask": torch.ones((1, 256)).to(model.device),
    "pixel_values": torch.randn(1, 3, 336, 336).to(model.device)
    }
    with torch.no_grad():
        try:
            flops = FlopCountAnalysis(model, dummy_inputs)
            print("Parameters:")
            print(parameter_count_table(model))
            print(f"FLOPs: {flops.total()/1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"계산 실패?: {e}")

    trainer.train()
    merged_model = model.merge_and_unload()   # FP16/full‑precision 가정
    
    # ② 저장 경로 지정 (예: outputs/merged_final)
    save_dir = os.path.join(training_args.output_dir, "merged_final")
    os.makedirs(save_dir, exist_ok=True)
    
    # ③ 모델 + 토크나이저 + 프로세서 저장
    merged_model.save_pretrained(save_dir, safe_serialization=True)     # safetensors
    language_processor.save_pretrained(save_dir)                       # tokenizer_config.json …
    vision_processor.save_pretrained(save_dir)      
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
