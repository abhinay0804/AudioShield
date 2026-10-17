"""
PyTorch Dataset module for ASVspoof 2019 LA dataset.
Handles protocol parsing, audio loading, resampling, silence trimming, peak normalization, and fixed-length padding/truncation.
Includes a dummy dataset generator for initial testing and demonstration.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf
from scipy.io import wavfile

from src.config import config, PathConfig, AudioConfig


def generate_dummy_dataset(
    output_dir: Path = config.paths.dummy_data_dir,
    num_train: int = 100,
    num_dev: int = 30,
    num_eval: int = 30,
    sample_rate: int = 16000,
    duration_sec: float = 4.0
) -> Dict[str, Path]:
    """
    Generates a synthetic dummy dataset mimicking ASVspoof 2019 LA layout.
    Useful for verification, testing, and execution when full dataset is not downloaded yet.
    """
    output_dir = Path(output_dir)
    la_dir = output_dir / "LA"
    proto_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": (num_train, la_dir / "ASVspoof2019_LA_train" / "flac", proto_dir / "ASVspoof2019.LA.cm.train.trl.txt"),
        "dev": (num_dev, la_dir / "ASVspoof2019_LA_dev" / "flac", proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"),
        "eval": (num_eval, la_dir / "ASVspoof2019_LA_eval" / "flac", proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt")
    }
    
    attack_types_spoof = [f"A{i:02d}" for i in range(7, 20)]  # A07 to A19
    rng = np.random.default_rng(seed=42)
    created_paths = {}
    
    for split_name, (num_samples, audio_dir, proto_file) in splits.items():
        audio_dir.mkdir(parents=True, exist_ok=True)
        proto_lines = []
        
        for i in range(num_samples):
            utt_id = f"LA_{split_name[0].upper()}_{i:07d}"
            speaker_id = f"LA_{1000 + i % 20:04d}"
            is_bonafide = (i % 3 == 0)  # ~33% bonafide, 67% spoof
            
            if is_bonafide:
                attack_type = "-"
                key = "bonafide"
                # Synthetic bonafide audio: harmonic sine wave series + subtle noise
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
                audio = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
                audio += 0.01 * rng.standard_normal(len(t))
            else:
                attack_type = rng.choice(attack_types_spoof)
                key = "spoof"
                # Synthetic spoofed audio: phase-disrupted tone + higher high-freq noise (simulating vocoder artifacts)
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
                audio = 0.4 * np.sin(2 * np.pi * 440 * t + np.random.uniform(0, np.pi, len(t)))
                audio += 0.05 * rng.standard_normal(len(t))
            
            audio = audio / (np.max(np.abs(audio)) + 1e-8)  # normalize
            flac_path = audio_dir / f"{utt_id}.flac"
            
            # Save audio as FLAC
            sf.write(str(flac_path), audio.astype(np.float32), sample_rate, format="FLAC")
            
            # Line format: SPEAKER_ID UTTERANCE_ID SYSTEM_ID ATTACK_TYPE KEY
            proto_lines.append(f"{speaker_id} {utt_id} - {attack_type} {key}\n")
        
        with open(proto_file, "w") as f:
            f.writelines(proto_lines)
            
        created_paths[split_name] = proto_file
        
    return created_paths


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 Logical Access (LA) dataset.
    
    Yields:
        waveform: Tensor of shape [1, target_samples] (16kHz audio)
        label: Tensor containing class index (1 = Bonafide, 0 = Spoof)
        attack_type: str identifier ("bonafide", "A07"..."A19")
        utt_id: str utterance ID (e.g. "LA_E_2834744")
    """
    def __init__(
        self,
        protocol_path: Union[str, Path],
        audio_dir: Union[str, Path],
        audio_cfg: AudioConfig = config.audio,
        is_train: bool = True,
        auto_generate_dummy: bool = False
    ):
        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)
        self.audio_cfg = audio_cfg
        self.is_train = is_train
        
        # Check if files exist; synthetic data is strictly disabled by default for training/evaluation
        if not self.protocol_path.exists() or not self.audio_dir.exists():
            if auto_generate_dummy:
                print(f"[Dataset WARNING] Real dataset files not found at {self.protocol_path}. Generating dummy dataset for unit testing only...")
                generate_dummy_dataset()
                # Update paths to dummy dataset locations
                split = "train" if "train" in str(protocol_path).lower() else ("dev" if "dev" in str(protocol_path).lower() else "eval")
                self.protocol_path = config.paths.dummy_data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.trl.txt"
                self.audio_dir = config.paths.dummy_data_dir / "LA" / f"ASVspoof2019_LA_{split}" / "flac"
            else:
                raise FileNotFoundError(
                    f"Real ASVspoof 2019 LA dataset not found at protocol path: '{self.protocol_path}' or audio dir: '{self.audio_dir}'. "
                    f"SYNTHETIC DATA FALLBACK IS STRICTLY DISABLED for training/evaluation per project policy. "
                    f"Please ensure the official ASVspoof 2019 LA dataset is extracted to data/LA/."
                )

        self.samples = self._parse_protocol()
        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.protocol_path.name}")

    def _parse_protocol(self) -> List[Dict[str, Union[str, int]]]:
        """
        Parses protocol TXT file line by line.
        Format: SPEAKER_ID UTTERANCE_ID SYSTEM_ID ATTACK_TYPE KEY
        Example: LA_0079 LA_E_2834744 - - bonafide
                 LA_0079 LA_E_2550189 - A07 spoof
        """
        samples = []
        with open(self.protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                speaker_id, utt_id, sys_id, attack_type, key = parts[:5]
                label = 1 if key.lower() == "bonafide" else 0
                attack_type_str = "bonafide" if key.lower() == "bonafide" else attack_type
                
                samples.append({
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "attack_type": attack_type_str,
                    "label": label,
                    "audio_path": self.audio_dir / f"{utt_id}.flac"
                })
        return samples

    def _trim_silence(self, waveform: torch.Tensor, top_db: float = 30.0) -> torch.Tensor:
        """Trims leading and trailing silent frames using energy thresholding."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Calculate frame energy
        energy = torch.mean(waveform ** 2, dim=0)
        max_energy = torch.max(energy)
        if max_energy == 0:
            return waveform
            
        threshold = max_energy * (10 ** (-top_db / 10.0))
        non_silent = torch.where(energy > threshold)[0]
        
        if len(non_silent) > 0:
            start_idx = non_silent[0].item()
            end_idx = non_silent[-1].item() + 1
            return waveform[:, start_idx:end_idx]
        return waveform

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Peak amplitude normalization to [-1.0, 1.0]."""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            return waveform / max_val
        return waveform

    def _pad_or_crop(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pads (with zeros) or crops (center-crop for train, repeat for short audio) to target_length."""
        channels, length = waveform.shape
        if length == target_length:
            return waveform
        elif length < target_length:
            # Repeat waveform or pad zero
            num_repeats = (target_length // length) + 1
            repeated = waveform.repeat(1, num_repeats)
            return repeated[:, :target_length]
        else:
            # Crop to target length (random crop during training, center crop during eval)
            if self.is_train:
                max_start = length - target_length
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = (length - target_length) // 2
            return waveform[:, start:start + target_length]

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Loads flac audio, resamples to target 16kHz, trims silence, normalizes, and pads/crops."""
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception:
            # Fallback to soundfile if torchaudio reader fails
            data, sample_rate = sf.read(str(audio_path))
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to target rate if needed
        if sample_rate != self.audio_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.audio_cfg.sample_rate)
            waveform = resampler(waveform)

        # Trim silence if configured
        if self.audio_cfg.trim_silence:
            waveform = self._trim_silence(waveform, top_db=self.audio_cfg.top_db)

        # Peak normalize
        if self.audio_cfg.normalize:
            waveform = self._normalize(waveform)

        # Pad or crop to exact target samples
        waveform = self._pad_or_crop(waveform, self.audio_cfg.target_samples)
        return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        item = self.samples[idx]
        waveform = self.load_audio(item["audio_path"])
        label_tensor = torch.tensor(item["label"], dtype=torch.long)
        return waveform, label_tensor, item["attack_type"], item["utt_id"]


if __name__ == "__main__":
    print("Testing Dataset Module...")
    # Generate dummy dataset for self-contained test
    generate_dummy_dataset()
    dataset = ASVspoofDataset(
        protocol_path=config.paths.dummy_data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trl.txt",
        audio_dir=config.paths.dummy_data_dir / "LA" / "ASVspoof2019_LA_train" / "flac",
        is_train=True
    )
    print(f"Dataset size: {len(dataset)}")
    wf, lbl, attack, utt = dataset[0]
    print(f"Sample 0 -> Waveform shape: {wf.shape}, Label: {lbl.item()} ({'Bonafide' if lbl.item()==1 else 'Spoof'}), Attack: {attack}, Utt: {utt}")
