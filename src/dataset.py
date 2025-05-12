import torch
import json
import os
from torch.utils.data import Dataset
from typing import Dict, List
from typing import Union
import numpy as np
from decord import VideoReader, cpu
from pathlib import Path

from transformers import PreTrainedTokenizer
from transformers import AutoProcessor
DEFAULT_IMAGE_TOKEN = "<image>"
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    video_path = str(video_path)
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

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
        """
        VLM Dataset for training and evaluation.

        Args:
            data_path (str): Path to the dataset file.
            tokenizer: Tokenizer for encoding text.
            image_placeholder (str): Placeholder for images in the text.
            force_sample (bool): Whether to force uniform frame sampling.
        """
        root = Path(data_path)
        # --- JSON 파일 경로 수집 ------------------------------------------------
        if data_files is None:
            json_paths = sorted(root.rglob("*.json"))           # ①
        elif isinstance(data_files, str):
            candidate = root / data_files
            if candidate.is_dir():                              # ②
                json_paths = sorted(candidate.rglob("*.json"))
            elif candidate.suffix == ".json":                   # ③
                json_paths = [candidate]
            else:
                raise ValueError(f"data_files 경로가 올바르지 않습니다: {candidate}")
        elif isinstance(data_files, (list, tuple)):             # ④
            json_paths = [root / p for p in data_files]
        else:
            raise TypeError("data_files는 str, list[str], None만 허용됩니다.")

        if not json_paths:
            raise FileNotFoundError("주어진 경로에서 JSON 파일을 찾지 못했습니다.")

        # --- JSON 로드 & 병합 ---------------------------------------------------
        self.data = []
        for p in json_paths:
            with open(p, "r") as f:
                self.data.extend(json.load(f))   # 각 파일이 list[dict] 구조라고 가정

        self.data_path = data_path
        self.max_frames_num = max_frames_num
        self.fps = fps
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.image_placeholder = image_placeholder
        self.force_sample = force_sample
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor | Dict]:
        """
        Returns a dictionary containing:
            input_ids:       (seq_len,)          – token ids for the LLM
            attention_mask:  (seq_len,)          – attention mask for the LLM
            pixel_values:    (num_frames, C, H, W) – processed video frames for the vision encoder
            video_path:      str                 – path to the raw video (optional downstream use)
        """
        # ---------- Video ----------
        video_path = Path(self.data_path) / self.data[index]["video"]
        spare_frames, frame_time, video_time = self.get_video_samples(video_path)

        # Process frames with the vision/image processor
        if self.img_processor is None:
            raise ValueError("`img_processor` must be provided.")
        pixel_values = self.img_processor(
            images=list(spare_frames), return_tensors="pt"
        )["pixel_values"]  # shape: (num_frames, C, H, W)

        # ---------- Text ----------
        conversations = self.data[index]["conversations"]

        # 1) system message with video meta‑data
        system_instruction = ("You are a helpful assistant."
            f"Video length: {video_time:.2f}s. "
            f"Selected frame timestamps: {frame_time}."
        )
        messages = [{"role": "system", "content": system_instruction}]

        # 2) user ↔ assistant turns
        for convo in conversations:
            role = "user" if convo["from"] == "human" else "assistant"
            messages.append({"role": role, "content": convo["value"]})

        # Tokenize with HF chat template
        if self.tokenizer is None:
            raise ValueError("`tokenizer` must be provided.")
        for i in messages:
            print(i)
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_assistant_tokens_mask = True,
            return_dict=True,
        )
        input_ids =  tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "video_path": str(video_path),
        }
    """
    'conversations': [{'from': 'human', 'value': '<image>\nWhy is the blue sweater guy looking at the shirtless men?\nOptions:\nA. sharing with his friends.\nB. found the man funny.\nC. poor vision.\nD. keep hands warm.\nE. training.\nPlease provide your answer by stating the letter followed by the full option.'}, {'from': 'gpt', 'value': 'E. training.'}]
    """
    def get_video_samples(self, video_path):
        return load_video(
            video_path,
            self.max_frames_num,
            fps=self.fps,
            force_sample=self.force_sample,
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