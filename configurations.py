import torch
train_config = {
    "batch_size": 2,
    "num_epochs": 8,
    "learning_rate": 3e-5,
    "weight_decay": 1e-5,
    "val_size": 0.2,
    "random_state": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "pretrained_model_name": "facebook/wav2vec2-base-960h",
    "train_audio_dir": "/kaggle/input/shl-intern-hiring-assessment/dataset/audios_train",
    "train_metadata_path": "/kaggle/input/shl-intern-hiring-assessment/dataset/train.csv",
    "max_audio_length": 1000000, 
    "use_wandb": True,
    "project_name": "wav2vec2-grammar"
}

inference_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "pretrained_model_name": "facebook/wav2vec2-base-960h",
    "sampling_rate": 16000,
    "model_path": "best_model.pt",
    "test_metadata_path": "./dataset/test.csv",
    "batch_size": 8,
    "test_audio_dir": "./dataset/audios_test",
    "max_audio_length": 1000000  
}