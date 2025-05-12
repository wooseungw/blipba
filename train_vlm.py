import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoProcessor,
)
from peft import LoraConfig, TaskType, get_peft_model

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

# ---------------------------------------------------------------------------- #
# 1. Dataset for Images & Video Frames
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# 2. Collator
# ---------------------------------------------------------------------------- #
class MultimodalCollator:
    def __init__(self, data_collator=default_data_collator):
        self.data_collator = data_collator

    def __call__(self, features):
        # Extract all components from features
        batch = {
            'pixel_values': torch.stack([f['pixel_values'] for f in features]),
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
def get_args():
    parser = argparse.ArgumentParser(description="Train Custom VLM with images and videos")
    # Backbone models
    parser.add_argument("--vision_model_name", type=str, default="facebook/dino-vitb16")
    parser.add_argument("--language_model_name", type=str, default="gpt2")
    # Resampler and pooling
    parser.add_argument("--use_resampler", action="store_true")
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_patch_merge_type", type=str, default="maxpool2x2")
    parser.add_argument("--max_num_patches", type=int, default=None)
    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules", nargs='+', default=["q_proj","k_proj","v_proj","o_proj"]
    )
    # Training hyperparams
    parser.add_argument("--output_dir", type=str, default="./vlm_output")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    # Data parameters
    parser.add_argument("--video_frame_count", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------- #
# 4. Main training flow
# ---------------------------------------------------------------------------- #
def main():
    args = get_args()
    # Config & Processor
    cfg = VisionLanguageConfig(
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        use_resampler=args.use_resampler,
        mm_spatial_pool_mode=args.mm_spatial_pool_mode,
        mm_patch_merge_type=args.mm_patch_merge_type,
        max_num_patches=args.max_num_patches,
    )
    processor = AutoProcessor.from_pretrained(cfg.vision_model_name)

    # Model + LoRA
    model = CustomVLMModel(cfg, vision_dtype=torch.float16, llm_dtype=torch.float16)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)

    # Dataset
    samples = []  # Replace with your data loading logic
    train_ds = MultimodalDataset(samples, processor, video_frame_count=args.video_frame_count)

    # Trainer setup
    collator = MultimodalCollator()
    training_args = TrainingArguments(
        output_dir=args['training']['output_dir'],
        run_name=args['training']['run_name'],
        num_train_epochs=args['training']['num_epochs'],
        per_device_train_batch_size=args['training']['batch_size']['train'],
        per_device_eval_batch_size=args['training']['batch_size']['eval'],
        gradient_accumulation_steps=args['training'].get('gradient_accumulation_steps', 4),
        gradient_checkpointing=args['training'].get('gradient_checkpointing', True),
        learning_rate=float(args['training'].get('learning_rate', 2e-5)),
        warmup_ratio=args['training'].get('warmup_ratio', 0.1),
        weight_decay=args['training'].get('weight_decay', 0.01),
        max_grad_norm=args['training'].get('max_grad_norm', 1.0),
        dataloader_num_workers=args['training'].get('dataloader_num_workers', 0),
        logging_dir=args['training']['logging_dir'],
        logging_steps=args['training']['logging_steps'],
        eval_strategy=args['training']['eval_strategy'],
        eval_steps=args['training'].get('eval_steps', 500),
        save_strategy=args['training']['save_strategy'],
        save_steps=args['training'].get('save_steps', 500),
        save_total_limit=args['training'].get('save_total_limit', 3),
        save_optimizer=False,      # skip saving optimizer state to reduce checkpoint size
        save_scheduler=False,      # skip saving scheduler state to reduce checkpoint size
        load_best_model_at_end=args['training']['load_best_model_at_end'],
        metric_for_best_model=args['training'].get('metric_for_best_model', 'eval_loss'),
        greater_is_better=args['training'].get('greater_is_better', False),
        fp16=True,  # DeepSpeed config에서 관리
        deepspeed=args['deepspeed']['config'] if args['deepspeed']['enabled'] else None,
        report_to=args['training']['report_to'],
        save_only_model=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
