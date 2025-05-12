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
from transformers import TrainerCallback
import yaml
from omegaconf import OmegaConf
from types import SimpleNamespace

from src.dataset import VLMDataset
from src.models.config import VisionLanguageConfig
from src.models.build import CustomVLMModel
from src.constant import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

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
    def __init__(self, vision_processor):
        self.vision_processor = vision_processor

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

# ---------------------------------------------------------------------------- #
# 2. Collator
# ---------------------------------------------------------------------------- #
class MultimodalCollator:
    def __init__(self, data_collator=default_data_collator):
        self.data_collator = data_collator

    def __call__(self, features):
        # Extract all components from features
        # Check if pixel_values is 5D (batch, frames, channels, height, width)
        # print("features[0]['pixel_values'].shape:", features[0]['pixel_values'].shape)
        _ ,c, h, w = features[0]['pixel_values'].shape
        pixel_values = torch.stack([f['pixel_values'] for f in features]).reshape(-1, c, h, w)
            
            
        batch = {
            'pixel_values': pixel_values,
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        }
        # Stack the labels and handle padding
        if 'labels' in features[0]:
            labels = torch.stack([f['labels'] for f in features])
            # Apply -100 masking to padding tokens in labels
            # This ensures padding tokens are ignored in the loss calculation
            padding_mask = batch['attention_mask'] == 0
            labels[padding_mask] = -100
            batch['labels'] = labels
        
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

    # Load the model
    model = CustomVLMModel(model_config)
    vision_processor = AutoProcessor.from_pretrained(cfg.model.vision_model_name)
    language_processor = deepcopy(model.tokenizer)

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
        tokenizer_name_or_path = cfg.model.llm_model_name,
        target_modules  = lora_target_modules,
        bias            = "none"
    )
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
        # Hugging Face 5.0 이후 `tokenizer` 인자가 폐지 → processor/토크나이저 모두 처리
        processing_class=language_processor,   # future‑proof
        label_names=["labels"]                 # suppress “No label_names …” warning
    )
    trainer.add_callback(CopyProcessorCallback(vision_processor))
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
