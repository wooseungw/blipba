import torch.nn as nn


class VisionLanguageModel(nn.Module):
    """
    A model that combines a vision encoder and a language model for multimodal tasks.
    """
    
    def __init__(
        self,
        vision_encoder_name: str,
        language_model_name: str,
        vision_hidden_size: Optional[int] = None,
        projection_dim: Optional[int] = None,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = False,
    ):
        super().__init__()
        
        # Load vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
        self.vision_processor = AutoProcessor.from_pretrained(vision_encoder_name)
        
        # Load language model
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None: