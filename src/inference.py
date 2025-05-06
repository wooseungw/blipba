def run_inference(input_data):
    from src.model_loader import ModelLoader
    from src.utils.helpers import preprocess_input, postprocess_output

    try:
        # Load the model
        model_loader = ModelLoader()
        model = model_loader.load_model()

        # Preprocess the input data
        processed_input = preprocess_input(input_data)

        # Run inference
        predictions = model(processed_input)

        # Postprocess the output
        output = postprocess_output(predictions)

        return output

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None