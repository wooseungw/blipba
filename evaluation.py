import os
import argparse
import torch
import warnings
import numpy as np
import json
import random
from tqdm import tqdm
from pathlib import Path
from decord import VideoReader, cpu
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import importlib.util

from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

# 경고 무시
warnings.filterwarnings("ignore")

# 상수 직접 정의
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def load_video(video_path: str, max_frames_num: int, fps: int = 1, force_sample: bool = False, img_processor = None) -> Tuple:
    """
    VLMDataset의 load_video 함수를 복제한 버전
    비디오 파일에서 프레임 추출
    """
    video_path = str(video_path)
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    N, H, W, C = spare_frames.shape
    
    if N < max_frames_num:
        pad = np.zeros((max_frames_num - N, H, W, C), dtype=spare_frames.dtype)
        spare_frames = np.concatenate([spare_frames, pad], axis=0)
    elif N > max_frames_num:
        spare_frames = spare_frames[:max_frames_num]
    
    return spare_frames, frame_time, video_time

def import_src_modules():
    """
    src 디렉토리의 모듈들을 동적으로 import
    """
    # src 디렉토리 경로 찾기
    src_dir = None
    for parent_dir in ['.', '..', '../..']:
        if os.path.exists(os.path.join(parent_dir, 'src', 'models', 'config.py')):
            src_dir = os.path.abspath(parent_dir)
            break
    
    if not src_dir:
        raise ImportError("src 디렉토리를 찾을 수 없습니다. 올바른 작업 디렉토리에서 실행하고 있는지 확인하세요.")
    
    import sys
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 필요한 모듈 import
    from src.models.config import VisionLanguageConfig
    from src.models.build import CustomVLMModel
    
    return VisionLanguageConfig, CustomVLMModel

def load_lora_checkpoint(model_path):
    """
    LoRA 체크포인트 디렉토리에서 모델 로딩
    """
    print(f"LoRA 체크포인트 로딩 중: {model_path}")
    
    # src 모듈 import
    VisionLanguageConfig, CustomVLMModel = import_src_modules()
    
    try:
        # 1. adapter_config.json 파일 로드
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"adapter_config.json 파일을 찾을 수 없습니다: {adapter_config_path}")
        
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # 기본 모델 경로 확인
        base_model_name = adapter_config.get("base_model_name_or_path")
        print(f"기본 모델 참조: {base_model_name}")
        
        # 2. 토크나이저 로드 - 디렉토리에서 직접 로드
        print("토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(f"토크나이저 로드됨: 어휘 크기 = {len(tokenizer)}")
        
        # 3. 비전 프로세서 로드
        print("비전 프로세서 로드 중...")
        try:
            # preprocessor_config.json 파일이 있으면 해당 디렉토리에서 로드
            preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
            if os.path.exists(preprocessor_path):
                vision_processor = AutoProcessor.from_pretrained(model_path)
                print("비전 프로세서 로드됨")
            else:
                # 지정된 기본 비전 모델 사용
                vision_model_name = "facebook/dinov2-small"  # 기본값
                vision_processor = AutoProcessor.from_pretrained(vision_model_name)
                print(f"대체 프로세서 로드됨: {vision_model_name}")
        except Exception as e:
            print(f"비전 프로세서 로드 중 오류: {e}")
            vision_processor = AutoProcessor.from_pretrained("facebook/dinov2-small")
            print("기본 비전 프로세서 로드됨: facebook/dinov2-small")
        
        # 4. 모델 설정 및 기본 모델 생성
        # adapter_config에서 모델 구성 정보 추출 또는 기본값 사용
        config_dict = {
            "vision_model_name": "facebook/dinov2-small",  # 기본값
            "language_model_name": base_model_name or "Qwen/Qwen3-0.6B",
            "projector_type": "mlp2x_gelu",
            "use_resampler": True,
            "mm_spatial_pool_mode": "average",
            "mm_newline_position": "grid",
            "freeze_vision": True,
            "freeze_llm": True
        }
        
        # 상위 디렉토리에서 config.json 파일 찾기 시도
        parent_dir = os.path.dirname(model_path)
        parent_config_path = os.path.join(parent_dir, "config.json")
        if os.path.exists(parent_config_path):
            print(f"상위 디렉토리에서 config.json 발견. 설정 로드 중: {parent_config_path}")
            with open(parent_config_path, 'r') as f:
                parent_config = json.load(f)
                # 기본 설정을 상위 디렉토리 설정으로 업데이트
                for key, value in parent_config.items():
                    if key in config_dict:
                        config_dict[key] = value
        
        # 설정 객체 생성
        config = VisionLanguageConfig(**config_dict)
        
        # 기본 모델 생성
        print("기본 모델 생성 중...")
        base_model = CustomVLMModel(
            config=config, 
            tokenizer=tokenizer,
            vision_dtype=torch.float16, 
            llm_dtype=torch.float16
        )
        
        # 5. LoRA 어댑터 로드
        print("LoRA 어댑터 로드 중...")
        
        # safetensors 파일 확인
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, model_path)
                print("PEFT 모델로 어댑터 로드 완료")
            except Exception as peft_e:
                print(f"PEFT 로드 중 오류: {peft_e}. 직접 가중치 로드 시도...")
                
                # 직접 어댑터 가중치 로드
                from safetensors.torch import load_file
                adapter_state_dict = load_file(adapter_path)
                base_model.load_state_dict(adapter_state_dict, strict=False)
                model = base_model
                print("safetensors에서 어댑터 가중치 직접 로드됨")
        else:
            # bin 파일 확인
            adapter_bin_path = os.path.join(model_path, "adapter_model.bin")
            if os.path.exists(adapter_bin_path):
                adapter_state_dict = torch.load(adapter_bin_path, map_location="cpu")
                base_model.load_state_dict(adapter_state_dict, strict=False)
                model = base_model
                print("adapter_model.bin에서 어댑터 가중치 로드됨")
            else:
                print("경고: 어댑터 가중치 파일을 찾을 수 없습니다. 기본 모델만 사용합니다.")
                model = base_model
        
        print("모델 로드 완료!")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise

    # GPU 사용 가능하면 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, vision_processor, device

def format_conversations(system_instruction, user_text, answer=None):
    """
    대화 형식으로 메시지 포맷팅
    VLMDataset의 __getitem__ 메소드 참고
    """
    conversations = [
        {"role": "system", "content": system_instruction}
    ]
    
    # 사용자 메시지 추가
    conversations.append({"role": "user", "content": user_text})
    
    # 답변이 있는 경우 어시스턴트 메시지도 추가
    if answer:
        conversations.append({"role": "assistant", "content": answer})
    
    return conversations

def load_dataset(data_path, data_root, max_ratio=1.0):
    """
    VLMDataset 스타일로 데이터셋 로드
    """
    print(f"데이터셋 로드 중: {data_path}")
    root = Path(data_root)
    
    # JSON 파일 로드
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"데이터셋 크기: {len(data)}")
    
    # 유효한 비디오만 필터링
    valid_data = []
    for entry in data:
        video_rel = entry.get("video", "")
        video_full = root / video_rel
        if video_full.is_file():
            valid_data.append(entry)
    
    print(f"유효한 비디오 수: {len(valid_data)}")
    
    # 최대 비율만큼만 사용
    if max_ratio < 1.0:
        random.shuffle(valid_data)
        num_samples = int(len(valid_data) * max_ratio)
        valid_data = valid_data[:num_samples]
        print(f"평가에 사용할 샘플 수: {len(valid_data)}")
    
    return valid_data

def prepare_sample_for_evaluation(sample, video_path, args, img_processor, tokenizer, device):
    """
    평가를 위한 샘플 준비 (VLMDataset.__getitem__ 참고)
    """
    # 비디오 로드
    spare_frames, frame_time, video_time = load_video(
        video_path, 
        args.max_frames_num, 
        fps=args.fps, 
        force_sample=args.force_sample,
        img_processor=img_processor
    )
    
    # 이미지 처리
    pixel_values = img_processor(
        images=list(spare_frames), 
        return_tensors="pt"
    )["pixel_values"].to(device).half()
    
    # 시스템 지시문 생성
    system_instruction = (
        "당신은 도움이 되는 비디오 분석 어시스턴트입니다."
        f" 비디오 길이: {video_time:.2f}초. "
        f"선택된 프레임 타임스탬프: {frame_time}."
    )
    
    # 질문 구성
    if args.dataset_name == "VideoMME" or "candidates" in sample:
        # 객관식 질문 처리
        question = sample["question"] + "\n"
        if "candidates" in sample:
            for idx, option in enumerate(sample["candidates"]):
                prefix = chr(65 + idx)  # A, B, C, D...
                question += f"{prefix}. {option}\n"
            question += "주어진 선택지 중 가장 적절한 답안은 무엇인가요? (알파벳만 답해주세요)"
    elif args.dataset_name == "VSI":
        # VSI 데이터셋 질문 처리
        question = sample["question"]
    elif args.dataset_name == "MovieChat":
        # MovieChat 데이터셋 질문 처리
        if 'time' in sample:
            timestamp = round(sample['time']/sample['fps'], 2)
            question = f"{timestamp}초 시점에서, {sample['question']}"
        else:
            question = sample["question"]
    else:
        # 기본 질문 처리
        question = sample["question"]
    
    # 대화 형식으로 변환
    messages = format_conversations(system_instruction, question)
    
    # 토크나이저 적용
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        return_tensors=None,
        enable_thinking=False,
    )
    
    # 토큰화
    tokenized = tokenizer(chat_text, return_tensors="pt").to(device)
    
    # 이미지 토큰 처리 (필요한 경우)
    if DEFAULT_IMAGE_TOKEN in chat_text:
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        
        # <image> 토큰을 IMAGE_TOKEN_INDEX로 변환
        mask = (input_ids == image_token_id)
        input_ids = input_ids.clone()
        input_ids[mask] = IMAGE_TOKEN_INDEX
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "frame_time": frame_time,
            "video_time": video_time
        }
    else:
        # 이미지 토큰이 없는 경우 프롬프트에 추가
        # 먼저 기존 텍스트 토큰화
        tokenized = tokenizer(chat_text, return_tensors="pt").to(device)
        
        # 이미지 토큰을 첫 번째 위치에 추가하는 프롬프트 다시 생성
        new_question = f"{DEFAULT_IMAGE_TOKEN} {question}"
        new_messages = format_conversations(system_instruction, new_question)
        new_chat_text = tokenizer.apply_chat_template(
            new_messages, 
            tokenize=False, 
            return_tensors=None,
            enable_thinking=False,
        )
        
        # 새 프롬프트 토큰화
        new_tokenized = tokenizer(new_chat_text, return_tensors="pt").to(device)
        input_ids = new_tokenized.input_ids
        attention_mask = new_tokenized.attention_mask
        
        # <image> 토큰을 IMAGE_TOKEN_INDEX로 변환
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        mask = (input_ids == image_token_id)
        input_ids = input_ids.clone()
        input_ids[mask] = IMAGE_TOKEN_INDEX
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "frame_time": frame_time,
            "video_time": video_time
        }

def evaluate_single_sample(model, sample, prepared_inputs, tokenizer):
    """
    단일 샘플에 대한 평가 수행
    """
    with torch.no_grad():
        try:
            outputs = model.generate(
                pixel_values=prepared_inputs["pixel_values"],
                input_ids=prepared_inputs["input_ids"],
                attention_mask=prepared_inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
            )
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 입력 프롬프트 디코딩
            prompt_text = tokenizer.batch_decode(prepared_inputs["input_ids"], skip_special_tokens=True)[0]
            
            # 프롬프트 부분 제거하여 실제 생성된 답변만 추출
            if generated_text.startswith(prompt_text):
                answer_text = generated_text[len(prompt_text):].strip()
            else:
                # 단순 접두어 매칭이 실패한 경우, 어시스턴트 응답 부분만 추출 시도
                assistant_marker = "어시스턴트:"
                if assistant_marker in generated_text:
                    answer_text = generated_text.split(assistant_marker, 1)[1].strip()
                else:
                    answer_text = generated_text.strip()
            
            return answer_text
            
        except Exception as e:
            print(f"샘플 평가 중 오류 발생: {e}")
            return f"ERROR: {str(e)}"

def fuzzy_matching(pred):
    """
    답변의 첫 단어만 추출하여 간단한 퍼지 매칭 수행
    """
    if not pred:
        return ""
    
    # 객관식 문자(A~E) 추출 시도
    for char in pred:
        if char.upper() in ["A", "B", "C", "D", "E"]:
            return char.upper()
    
    # 실패 시 첫 단어 반환
    return pred.split(' ')[0].rstrip('.').strip()

def calculate_metrics(result_list):
    """
    평가 결과에 대한 지표 계산
    """
    results = {"all": {"correct": 0, "total": 0}}
    
    for sample in result_list:
        if "answer" not in sample:
            continue
            
        results["all"]["total"] += 1
        
        # 질문 유형별 결과 추적
        if "question_type" in sample:
            q_type = sample["question_type"]
            if q_type not in results:
                results[q_type] = {"correct": 0, "total": 0}
            results[q_type]["total"] += 1
        
        # 정답 확인
        pred = fuzzy_matching(sample["prediction"]).lower()
        answer = sample["answer"].lower()
        
        if pred == answer:
            results["all"]["correct"] += 1
            if "question_type" in sample:
                results[sample["question_type"]]["correct"] += 1
    
    # 정확도 계산
    for key in results:
        if results[key]["total"] > 0:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]
        else:
            results[key]["accuracy"] = 0.0
    
    return results

def main():
    parser = argparse.ArgumentParser(description="LoRA 체크포인트를 활용한 VLM 모델 평가")

    # 모델 관련 인자
    parser.add_argument("--model_path", type=str, required=True,
                       help="평가할 LoRA 체크포인트 디렉토리 경로")
    parser.add_argument("--max_frames_num", type=int, default=64,
                       help="비디오에서 추출할 최대 프레임 수")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="생성할 최대 토큰 수")
    parser.add_argument("--fps", type=int, default=1,
                       help="프레임 추출 fps")
    
    # 데이터셋 관련 인자
    parser.add_argument("--dataset_name", type=str, default="VideoMME",
                       help="평가할 데이터셋 이름 (VideoMME, VSI, MovieChat 등)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="데이터셋 JSON 파일 경로")
    parser.add_argument("--video_root", type=str, required=True,
                       help="비디오 파일이 저장된 루트 디렉토리")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="결과를 저장할 디렉토리")
    parser.add_argument("--test_ratio", type=float, default=1.0,
                       help="평가에 사용할 데이터셋 비율 (0.0 ~ 1.0)")
    
    # 기타 옵션
    parser.add_argument("--force_sample", action="store_true",
                       help="항상 균일한 샘플링 사용")
    parser.add_argument("--force_reevaluate", action="store_true",
                       help="이미 평가된 샘플도 다시 평가할지 여부")
    parser.add_argument("--seed", type=int, default=42,
                       help="난수 생성기 시드")
    
    args = parser.parse_args()
    
    # 난수 시드 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "outputs"), exist_ok=True)
    
    # 모델, 토크나이저, 프로세서 로드
    model, tokenizer, vision_processor, device = load_lora_checkpoint(args.model_path)
    
    # 데이터셋 로드
    dataset = load_dataset(args.data_path, args.video_root, args.test_ratio)
    
    # 평가 실행
    result_list = []
    for idx, sample in enumerate(tqdm(dataset, desc="평가 진행 중")):
        # 결과가 이미 존재하는지 확인
        sample_save_path = Path(args.results_dir) / "outputs" / f"{sample['id']}.json"
        if sample_save_path.exists() and not args.force_reevaluate:
            try:
                with open(sample_save_path, 'r', encoding='utf-8') as f:
                    loaded_sample = json.load(f)
                    result_list.append(loaded_sample)
                    print(f"[{idx}] 기존 결과 로드: {loaded_sample.get('id')}")
                    if "answer" in loaded_sample:
                        print(f"정답: {loaded_sample['answer']}, 예측: {loaded_sample['prediction']}")
                    continue
            except Exception as e:
                print(f"기존 결과 로드 실패: {e}, 다시 평가합니다.")
        
        # 비디오 경로 구성
        video_path = Path(args.video_root) / sample["video"]
        
        # 평가를 위한 입력 준비
        prepared_inputs = prepare_sample_for_evaluation(
            sample, video_path, args, vision_processor, tokenizer, device
        )
        
        # 평가 수행
        prediction = evaluate_single_sample(model, sample, prepared_inputs, tokenizer)
        
        # 결과 저장
        result_sample = deepcopy(sample)
        result_sample["prediction"] = prediction
        result_list.append(result_sample)
        
        # 결과 파일로 저장
        with open(sample_save_path, 'w', encoding='utf-8') as f:
            json.dump(result_sample, f, indent=4, ensure_ascii=False)
        
        # 로그 출력
        print(f"[{idx}] 샘플 ID: {sample.get('id')}")
        if "answer" in sample:
            print(f"정답: {sample['answer']}, 예측: {prediction}")
        else:
            print(f"예측: {prediction}")
    
    # 평가 지표 계산
    metrics = calculate_metrics(result_list)
    print("\n평가 결과:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # 최종 결과 저장
    with open(os.path.join(args.results_dir, "results.json"), "w", encoding='utf-8') as f:
        json.dump({
            "metrics": metrics,
            "args": vars(args)
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n평가 완료! 결과가 {os.path.join(args.results_dir, 'results.json')}에 저장되었습니다.")

if __name__ == "__main__":
    main()