import argparse
import torch
import warnings
import numpy as np
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
import functools
import itertools
import random
from tqdm import tqdm
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from copy import deepcopy
from types import SimpleNamespace

from transformers import AutoProcessor, AutoTokenizer
from peft import PeftModel

from src.models.config import VisionLanguageConfig
from src.models.captionvlm import CaptioningVLM
from src.constant import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX

from decord import VideoReader, cpu

# 경고 무시
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """
    비디오를 로드하고 지정된 프레임 수로 샘플링합니다.
    
    Args:
        video_path: 비디오 파일 경로
        max_frames_num: 최대 프레임 수
        fps: 초당 프레임 수
        force_sample: 강제 샘플링 여부
        
    Returns:
        샘플링된 프레임, 프레임 시간, 비디오 시간
    """
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

def get_prompt(dataset_name, sample, video_time=None, num_frames=None, frame_time=None):
    """
    프롬프트 생성 함수
    
    Args:
        dataset_name: 데이터셋 이름
        sample: 현재 샘플
        video_time: 비디오 시간 (초)
        num_frames: 프레임 수
        frame_time: 프레임 시간 문자열
        
    Returns:
        생성된 프롬프트
    """
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    if dataset_name in ['VSI']:
        prompt += "These are frames of a video.\n"
        prompt += sample["question"] + "\n"
        if 'candidates' in sample:
            for op in sample["candidates"]:
                prompt += f"{op}\n"
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += "Please answer the question using a single word or phrase."
    elif dataset_name in ['MovieChat']:
        if video_time is None:
            prompt += "These are frames of a video.\n"
        if 'time' in sample:
            timestamp = round(sample['time'] / sample['fps'], 2)
            prompt += f"At time {timestamp}s, "
        prompt += sample["question"] + "\n"
        prompt += "Please answer the question using a single word, phrase, or sentence."
    else:
        options_letter = get_options_letter(len(sample['candidates']))
        prompt += f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {options_letter} of the correct option.\n"
        prompt += sample["question"] + "\n"
        for op in sample["candidates"]:
            prompt += f"{op}\n"
        prompt += f"The best answer is:"
        
    question = DEFAULT_IMAGE_TOKEN + prompt
    return question

def get_options_letter(len_options):
    """
    선택지 옵션 문자열 반환
    """
    if len_options == 2:
        return '(A or B)'
    elif len_options == 3:
        return '(A, B or C)'
    elif len_options == 4:
        return '(A, B, C or D)'
    elif len_options == 5:
        return '(A, B, C, D, or E)'
    else:
        raise NotImplementedError

def fuzzy_matching(pred):
    """
    예측 결과를 정제하는 함수
    """
    return pred.split(' ')[0].rstrip('.').strip()

def load_config(config_path: str):
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        로드된 설정 객체
    """
    cfg = OmegaConf.load(config_path)

    # 모델 이름 기반 템플릿 해석
    model_name = cfg.model.name
    if 'training' in cfg:
        cfg.training.output_dir = cfg.training.output_dir.format(model_name=model_name)
        cfg.training.run_name = cfg.training.run_name.format(model_name=model_name)
        cfg.training.logging_dir = cfg.training.logging_dir.format(model_name=model_name)
    
    # 하위 호환성: 'data' 섹션만 있는 설정 파일 처리
    if "dataset" not in cfg and "data" in cfg:
        cfg.dataset = cfg.data  # 하위 호환성 별칭
    
    return cfg

def run_evaluation(args):
    """
    주 평가 함수
    
    Args:
        args: 명령행 인자
        
    Returns:
        평가 결과 리스트
    """
    # CUDA 장치 설정
    torch.cuda.set_device(args.rank)
    
    # 데이터셋 로드
    print(f"데이터셋 로드 중: {args.data_path}")
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    
    # 데이터셋 샘플링 (테스트 비율에 따라)
    random.shuffle(dataset)
    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[args.rank:num_samples:args.world_size]
    print(f"전체 샘플 수: {num_samples}")
    print(f"랭크 {args.rank}의 샘플 수: {len(dataset)}")
    
    # 저장된 모델 config 확인 및 로드
    model_dir = Path(args.model_path).parent
    saved_config_path = model_dir / "config.json"
    
    if saved_config_path.exists():
        print(f"저장된 모델 설정 파일 사용: {saved_config_path}")
        with open(saved_config_path, "r") as f:
            model_cfg = json.load(f)
        
        # 필요한 모델 정보 추출
        vision_model_name = model_cfg.get("vision_model_name", "facebook/dinov2-base")
        llm_model_name = model_cfg.get("language_model_name", "Qwen/Qwen3-0.6B")
        projector_type = model_cfg.get("projector_type", "linear")
        use_resampler = model_cfg.get("use_resampler", False)
        mm_spatial_pool_mode = model_cfg.get("mm_spatial_pool_mode", "average")
        mm_newline_position = model_cfg.get("mm_newline_position", "grid")
        freeze_vision = model_cfg.get("freeze_vision", True)
        freeze_llm = model_cfg.get("freeze_llm", False)
    elif args.config:
        # 설정 파일이 없는 경우 외부 config 로드
        print(f"외부 설정 파일 사용: {args.config}")
        cfg = load_config(args.config)
        
        # 필요한 모델 정보 추출
        vision_model_name = cfg.model.vision_model_name
        llm_model_name = cfg.model.llm_model_name
        projector_type = cfg.model.projector_type
        use_resampler = getattr(cfg.model, "use_resampler", False)
        mm_spatial_pool_mode = cfg.model.mm_spatial_pool_mode
        mm_newline_position = getattr(cfg.model, "mm_newline_position", "grid")
        freeze_vision = cfg.model.freeze_vision
        freeze_llm = cfg.model.freeze_llm
    else:
        raise ValueError("모델 설정을 찾을 수 없습니다. --config 인자를 지정하거나 모델 폴더에 config.json 파일이 있어야 합니다.")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        use_fast=True
    )
    
    # 이미지 프로세서 로드
    image_processor = AutoProcessor.from_pretrained(
        vision_model_name,
        use_fast=True
    )
    
    # VLM 모델 설정
    model_config = VisionLanguageConfig(
        vision_model_name=vision_model_name,
        language_model_name=llm_model_name,
        projector_type=projector_type,
        use_resampler=use_resampler,
        mm_spatial_pool_mode=mm_spatial_pool_mode,
        mm_newline_position=mm_newline_position,
        freeze_vision=freeze_vision,
        freeze_llm=freeze_llm,
    )
    
    # CaptioningVLM 모델 로드
    model = CaptioningVLM(model_config, tokenizer=tokenizer)
    
    # 학습된 모델 가중치 로드
    if args.model_path:
        print(f"모델 가중치 로드 중: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # LoRA 적용 (있는 경우)
    if args.lora_alpha:
        from peft import LoraConfig, TaskType, get_peft_model
        
        # LoRA 타겟 모듈 설정
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj",
            "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # LoRA 설정
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            base_model_name_or_path=cfg.model.llm_model_name,
            target_modules=lora_target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA 적용됨")
    
    # GPU로 모델 이동
    model = model.to(torch.device(f"cuda:{args.rank}"))
    model.eval()
    
    # 결과 디렉토리 생성
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)
    
    # 평가 실행
    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        sample_save_path = f"{args.results_dir}/outputs/{sample['id']}.json"
        
        # 이미 처리된 샘플은 건너뛰기
        if os.path.exists(sample_save_path) and not args.overwrite:
            with open(sample_save_path, 'r') as f:
                sample = json.load(f)
        else:
            # 비디오 로드
            video_path = os.path.join(args.video_root, sample["video"])
            video, frame_time, video_time = load_video(
                video_path, 
                args.max_frames_num, 
                fps=1, 
                force_sample=True
            )
            
            # 이미지 프로세싱
            video = image_processor.preprocess(
                video, 
                return_tensors="pt"
            )["pixel_values"].cuda().to(torch.float16)
            
            # 프롬프트 생성
            if args.use_time_ins:
                prompt_question = get_prompt(
                    args.dataset_name, 
                    sample, 
                    video_time=video_time, 
                    num_frames=args.max_frames_num, 
                    frame_time=frame_time
                )
            else:
                prompt_question = get_prompt(args.dataset_name, sample)
            
            # 토큰화
            inputs = tokenizer(
                prompt_question, 
                return_tensors="pt"
            ).to(model.device)
            
            # 추론
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values=video,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=args.max_new_tokens,
                )
                
            # 결과 디코딩
            text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            sample["prediction"] = text_outputs
            
            # 결과 저장
            with open(sample_save_path, "w") as f:
                json.dump(sample, f, indent=4)
        
        result_list.append(sample)
        
        # 결과 출력
        if "answer" in sample:
            print(cnt, "GT:", sample["answer"], "Pred:", sample["prediction"])
        else:
            print(cnt, "Pred:", sample["prediction"])
    
    return result_list

def main():
    parser = argparse.ArgumentParser(description="CaptioningVLM 모델 평가")
    
    # 모델 관련 인자
    parser.add_argument("--config", type=str, default=None, help="선택적: 외부 설정 파일 경로 (모델 폴더에 config.json이 없는 경우)")
    parser.add_argument("--model_path", type=str, default="outputs/test/merged_final",required=True, help="평가할 모델 가중치 경로")
    parser.add_argument("--max_frames_num", type=int, default=64, help="최대 프레임 수")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="생성할 최대 토큰 수")
    parser.add_argument("--use_time_ins", action="store_true", help="시간 정보를 프롬프트에 포함할지 여부")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA 알파 값")
    
    # 데이터 관련 인자
    parser.add_argument("--dataset_name", type=str, default="VideoMME", help="데이터셋 이름")
    parser.add_argument("--data_path", type=str, required=True, help="데이터셋 파일 경로")
    parser.add_argument("--video_root", type=str, required=True, help="비디오 파일 경로")
    parser.add_argument("--results_dir", type=str, default="./results", help="결과 저장 디렉토리")
    parser.add_argument("--test_ratio", type=float, default=1.0, help="테스트 데이터 비율")
    
    # 평가 관련 인자
    parser.add_argument("--rank", type=int, default=0, help="분산 평가 시 랭크")
    parser.add_argument("--world_size", type=int, default=1, help="분산 평가 시 월드 사이즈")
    parser.add_argument("--multiprocess", action="store_true", help="멀티프로세스 사용 여부")
    parser.add_argument("--calc_acc", action="store_true", help="정확도 계산 여부")
    parser.add_argument("--overwrite", action="store_true", help="기존 결과 덮어쓰기 여부")
    
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)
    
    # 멀티프로세스 사용 여부에 따라 평가 실행
    if args.multiprocess:
        mp.set_start_method("spawn")
        print("멀티프로세스 평가 시작")
        n_gpus = torch.cuda.device_count()
        args.world_size = n_gpus
        print("사용 가능한 GPU 수:", args.world_size)
        
        with Pool(args.world_size) as pool:
            tasks = []
            for rank in range(args.world_size):
                args_copy = deepcopy(args)
                args_copy.rank = rank
                tasks.append(pool.apply_async(run_evaluation, (args_copy,)))
            
            result_lists = [task.get() for task in tasks]
        
        print("평가 완료")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run_evaluation(args)
    
    # 정확도 계산
    if args.calc_acc:
        results = {"all": {"correct": 0, "total": 0}}
        for sample in result_list:
            if "answer" not in sample:
                continue
            
            results["all"]["total"] += 1
            
            if "question_type" in sample:
                if sample["question_type"] not in results:
                    results[sample["question_type"]] = {"correct": 0, "total": 0}
                results[sample["question_type"]]["total"] += 1
            
            if sample["answer"].lower() == fuzzy_matching(sample["prediction"]).lower():
                results["all"]["correct"] += 1
                if "question_type" in sample:
                    results[sample["question_type"]]["correct"] += 1
        
        # 정확도 계산
        for key in results:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]
        
        print(results)
        
        # 결과 저장
        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()