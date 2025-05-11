import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from typing import Dict, Optional, Union , List
from projector import build_vision_projector
from abc import ABC

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
# vision encoder: ["google/siglip-so400m-patch14-384","openai/clip-vit-large-patch14"]

class VisionLanguageModel(ABC):
    """
    A model that combines a vision encoder and a language model for multimodal tasks.
    """
    def __init__(
        self,
        config,
        vision_cfg: Optional[str] = None,
        llm_cfg: Optional[str] = None,
        ):
        super().__init__()
        self.config = config
        self.vision_cfg = config.vision_config
        self.text_cfg = config.text_config
        print("Vision Encoder Config: ", self.vision_cfg)
        print("Language Model Config: ", self.text_cfg)
        
        # Load vision encoder
        self.vision_encoder = AutoModel.from_pretrained(self.vision_cfg)
        # self.vision_processor = AutoProcessor.from_pretrained(self.vision_cfg)
        # Load language model
        self.language_model = AutoModelForCausalLM.from_pretrained(self.text_cfg)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.text_cfg)
        
        self.projector = build_vision_projector(
            config=config,
            projector_type = config.projector_type,
            vision_cfg=self.vision_encoder.config,
        )
    
    def encode_image(self, pixel_values):
        """Encode image using vision encoder."""
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        projected_features = self.projection(image_features)
        return projected_features
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        **kwargs
    ):
        if pixel_values is not None and image_features is None:
            image_features = self.encode_image(pixel_values)
            
        # Prepare for language model input
        if image_features is not None:
            # Expand image features to match batch size of text
            batch_size = input_ids.shape[0]
            if image_features.shape[0] == 1 and batch_size > 1:
                image_features = image_features.expand(batch_size, -1)
                
            # Combine image features with input embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # Prepend image features to the sequence
            # Assuming first token position is used for image features
            inputs_embeds[:, 0, :] = image_features
            
            # Forward pass through language model
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # Text-only forward pass
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            
        return outputs

    

def build_vision_language_model(
    vision_encoder_name: str,
    language_model_name: str,
    config: Optional[Dict] = None,
) -> VisionLanguageModel:
    """
    Build and return a VisionLanguageModel.
    
    Args:
        vision_encoder_name: HuggingFace model ID for the vision encoder
        language_model_name: HuggingFace model ID for the language model
        config: Additional configuration parameters
        
    Returns:
        A VisionLanguageModel instance
    """
    if config is None:
        config = {}
        
    model = VisionLanguageModel(
        vision_encoder_name=vision_encoder_name,
        language_model_name=language_model_name,
        vision_hidden_size=config.get("vision_hidden_size"),
        projection_dim=config.get("projection_dim"),
        freeze_vision_encoder=config.get("freeze_vision_encoder", False),
        freeze_language_model=config.get("freeze_language_model", False),
    )
    
    return model