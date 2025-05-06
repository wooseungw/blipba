import argparse
from transformers import AutoModel, AutoTokenizer

def download_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face model hub.")
    parser.add_argument("model_name", type=str, help="The name of the model to download.")
    args = parser.parse_args()

    tokenizer, model = download_model(args.model_name)
    print(f"Model '{args.model_name}' downloaded successfully.")

if __name__ == "__main__":
    main()