from transformers import AutoTokenizer

# 토크나이저 로드 (예: Hugging Face에서 사전학습된 모델 사용)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 특수 토큰을 추가
special_tokens_dict = {
    "additional_special_tokens": ["<image>"]
}
tokenizer.add_special_tokens(special_tokens_dict)

def create_input_with_template(instruction, tokenizer):
    """
    주어진 instruction과 템플릿을 결합한 후 토큰화된 결과를 반환하는 함수.
    기대되는 입력은 [{'from': 'human','value': '<image>\nWh...},{}]
    Args:
    - instruction (str): 사용자가 입력한 지시문.
    - image_placeholder (str): <image>로 대체될 이미지 토큰. 기본값은 "<image>".

    Returns:
    - dict: 토크나이저에 의해 인코딩된 입력.
    """
    # 템플릿을 정의 (여기서는 단순 예시)
    
    
    # 템플릿과 instruction을 결합하여 토큰화
    inputs = tokenizer(template, return_tensors="pt")

    return inputs

# 예시 사용
instruction = "What is the object in the image? <image>"
encoded_input = create_input_with_template(instruction)

# 출력
print(encoded_input)
