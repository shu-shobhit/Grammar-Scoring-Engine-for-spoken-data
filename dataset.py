# dataset.py
import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

class AudioGrammarDataset(Dataset):
    def __init__(self, data_dir, metadata_path, is_test=False, max_length=1000000):
        self.df = pd.read_csv(metadata_path)
        self.audio_files = [
            os.path.join(data_dir, file) for file in self.df["filename"]
        ]
        self.is_test = is_test

        if not self.is_test:
            self.labels = self.df["label"]

        self.max_length = max_length  # Max length in samples (16kHz * 60 seconds = 960000)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        waveform, sample_rate = torchaudio.load(audio_path)

        waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        if waveform.shape[0] > self.max_length:
            waveform = waveform[: self.max_length]
            attention_mask = torch.ones(waveform.shape)
        else:
            padding = torch.zeros(self.max_length - waveform.shape[0])
            attention_mask = torch.ones(waveform.shape)
            waveform = torch.cat([waveform, padding])
            attention_mask = torch.cat([attention_mask, padding])
            
        if not self.is_test:
            label = self.labels[idx]

            return {
                "raw_waveform": waveform,
                "attention_mask": attention_mask,
                "label": torch.FloatTensor([label]),
            }
        else:
            return {"raw_waveform": waveform, "attention_mask": attention_mask}

def prepare_dataloaders(config):
    dataset = AudioGrammarDataset(
        data_dir=config["train_audio_dir"],
        metadata_path=config["train_metadata_path"],
        max_length=config["max_audio_length"]
    )

    train_size = int((1 - config["val_size"]) * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["random_state"])
    )

    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=False
    )

    print("Train and Val dataloaders prepared!")
    return train_loader, val_loader

def prepare_test_dataloader(config):
    test_dataset = AudioGrammarDataset(
        data_dir=config["test_audio_dir"],
        metadata_path=config["test_metadata_path"],
        is_test=True,
        max_length=config["max_audio_length"]
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"]
    )

    print("Test dataloader prepared!")
    return test_loader