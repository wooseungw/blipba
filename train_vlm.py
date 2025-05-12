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
from copy import deepcopy
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

def collator(features):
    return MultimodalCollator()(features)

# ---------------------------------------------------------------------------- #
# 3. Argument Parser
# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")
    return parser.parse_args()

# ---------------------------------------------------------------------------- #
# 4. Main training flow
# ---------------------------------------------------------------------------- #
def main():
    config = parse_args()
    # Config & Processor
    model_config = VisionLanguageConfig(
        vision_model_name=config.model.vision_model_name,
        language_model_name=config.model.language_model_name,
        projector_type=config.model.projector_type,
        use_resampler=config.model.use_resampler,
        mm_spatial_pool_mode=config.model.mm_spatial_pool_mode,
        mm_newline_position=config.model.mm_newline_position,
        freeze_vision=config.model.freeze_vision,
        freeze_llm=config.model.freeze_llm,
                                  )

    
    model = CustomVLMModel(model_config)
    vision_processor = AutoProcessor.from_pretrained(config.model.vision_model_name)
    language_processor = deepcopy(model.tokenizer)
    
    # Dataset
    train_ds = None #TODO
    
    # Training args
    model_name = config.model.name
    
    lora_config = LoraConfig(
    task_type=TaskType.SEQUENCE_CLASSIFICATION,
    r=128,  # rank
    lora_alpha=64,
    lora_dropout=0.1
    )
    
    training_args = TrainingArguments(
        output_dir=config.training,
        run_name=config.training.run_name,
        logging_dir=config.training.logging_dir,
        deepspeed=config['deepspeed']['config'] if config['deepspeed']['enabled'] else None,
        # Training
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        # Evaluation
        eval_strategy=config.training.eval_strategy,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        eval_steps=config.training.eval_steps,
        # Save
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        greater_is_better=config.training.greater_is_better,
        # Repoting
        report_to=config.training.report_to,
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
