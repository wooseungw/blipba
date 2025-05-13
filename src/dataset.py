import torch
import json
import os
from torch.utils.data import Dataset
from typing import Dict, List
from typing import Union
import numpy as np
from decord import VideoReader, cpu
from pathlib import Path
import logging
from PIL import Image

from transformers import PreTrainedTokenizer
from transformers import AutoProcessor
DEFAULT_IMAGE_TOKEN = "<image>"
def load_video(video_path, max_frames_num, fps=1, force_sample=False, img_processor=None):
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
    # --- Pad or truncate to fixed number of frames -------------------------
    N, H, W, C = spare_frames.shape
    if N < max_frames_num:
        pad = np.zeros((max_frames_num - N, H, W, C), dtype=spare_frames.dtype)
        spare_frames = np.concatenate([spare_frames, pad], axis=0)
    elif N > max_frames_num:
        spare_frames = spare_frames[:max_frames_num]
    return spare_frames, frame_time, video_time

def load_video(video_path, max_frames_num, fps=1, force_sample=False, img_processor=None):
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

class VLMDataset(Dataset):
    def __init__(
        self,
        data_path: str = "DATAS/train/",
        data_files: Union[str, List[str], None] = None,
        image_placeholder: str = DEFAULT_IMAGE_TOKEN,
        max_frames_num: int = 64,
        fps: int = 1,
        img_processor: AutoProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        force_sample: bool = False,
    ):
        root = Path(data_path)
        print(f"Loading data from {root}")
        if data_files is None:
            json_paths = sorted(root.rglob("*.json"))
        elif isinstance(data_files, str):
            candidate = root / data_files
            if candidate.is_dir():
                json_paths = sorted(candidate.rglob("*.json"))
            elif candidate.suffix == ".json":
                json_paths = [candidate]
            else:
                raise ValueError(f"data_files 경로가 올바르지 않습니다: {candidate}")
        elif isinstance(data_files, (list, tuple)):
            json_paths = [root / p for p in data_files]
        else:
            raise TypeError("data_files는 str, list[str], None만 허용됩니다.")

        if not json_paths:
            raise FileNotFoundError("주어진 경로에서 JSON 파일을 찾지 못했습니다.")

        self.data = []
        for p in json_paths:
            print(f"Loading {p}", end=" ")
            with open(p, "r") as f:
                self.data.extend(json.load(f))
        print()
        print("length of dataset:", len(self.data))

        valid_data = []
        for entry in self.data:
            video_rel = entry.get("video", "")
            video_full = Path(data_path) / video_rel
            if video_full.is_file():
                valid_data.append(entry)
        self.data = valid_data
        self.data_path = data_path
        self.max_frames_num = max_frames_num
        self.fps = fps
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.image_placeholder = image_placeholder
        self.force_sample = force_sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, Dict]]:
        video_path = Path(self.data_path) / self.data[index]["video"]
        spare_frames, frame_time, video_time = self.get_video_samples(video_path)

        if self.img_processor is None:
            raise ValueError("`img_processor` must be provided.")
        pixel_values = self.img_processor(
            images=list(spare_frames), return_tensors="pt"
        )["pixel_values"]

        conversations = self.data[index]["conversations"]
        system_instruction = (
            "You are a helpful assistant."
            f" Video length: {video_time:.2f}s. "
            f"Selected frame timestamps: {frame_time}."
        )
        messages = [{"role": "system", "content": system_instruction}]
        for convo in conversations:
            role = "user" if convo["from"] == "human" else "assistant"
            messages.append({"role": role, "content": convo["value"]})

        if self.tokenizer is None:
            raise ValueError("`tokenizer` must be provided.")
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        assistant_mask = tokenized["assistant_masks"].squeeze(0)

        # ✅ 학습 라벨은 assistant 응답만 학습, 나머지는 -100
        labels = input_ids.clone()
        labels[assistant_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "video_path": str(video_path),
        }

    def get_video_samples(self, video_path):
        return load_video(
            video_path,
            self.max_frames_num,
            fps=self.fps,
            force_sample=self.force_sample,
            img_processor=self.img_processor,
        )
        
if __name__ == "__main__":
    # Example usage
    from transformers import AutoProcessor
    from transformers import AutoTokenizer
    
    # Load the image processor and tokenizer
    processor = AutoProcessor.from_pretrained("facebook/dino-vitb16")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    dataset = VLMDataset(data_path="DATAS/train/",data_files="sample/sample.json", img_processor=processor, tokenizer=tokenizer)
    
    
    print(len(dataset))  # Print the number of entries in the dataset
    sample = dataset[0]
    print(sample.keys() ) # Access the first entry