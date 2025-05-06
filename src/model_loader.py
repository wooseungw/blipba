class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def get_model_info(self):
        if self.model is not None:
            return {
                "model_name": self.model_name,
                "model_type": type(self.model).__name__,
                "tokenizer_type": type(self.tokenizer).__name__,
            }
        else:
            return "Model not loaded."