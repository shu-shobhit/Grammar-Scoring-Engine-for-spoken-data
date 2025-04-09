import torch
import argparse
import os
from trainer import Trainer
from inference import Inference
from configurations import train_config, inference_config
from model import Wav2Vec2GrammarScoring

def setup_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def train_model(config):
    
    print(f"Starting training with device: {config['device']}")
    print(f"Using pretrained model: {config['pretrained_model_name']}")
    
    model = None
    if config.get("resume_from_checkpoint"):
        checkpoint_path = config["resume_from_checkpoint"]
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = Wav2Vec2GrammarScoring(config["pretrained_model_name"])
            model.load_state_dict(torch.load(checkpoint_path))
    

    trainer = Trainer(config=config, model=model)
    history, model = trainer.train()
    
    print(f"Training complete. Best validation RMSE: {trainer.best_val_rmse:.4f}")
    return model

def run_inference(config):
    
    print(f"Starting inference with device: {config['device']}")
    print(f"Using model from: {config['model_path']}")
    
    inference_engine = Inference(config=config)
    

    if config.get("output_path"):
        results = inference_engine.save_predictions(config["output_path"])
    else:
        scores = inference_engine.predict()
        print(f"Generated {len(scores)} predictions")
    
    return inference_engine

def main():
    parser = argparse.ArgumentParser(description="Grammar Scoring Model")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "inference"], 
        default="train",
        help="Run mode: 'train' or 'inference'"
    )
    parser.add_argument(
        "--config_override", 
        type=str, 
        default=None,
        help="Path to a JSON file with config overrides"
    )
    
    args = parser.parse_args()
    

    if args.config_override:
        import json
        with open(args.config_override, 'r') as f:
            overrides = json.load(f)
            
        if args.mode == "train":
            for k, v in overrides.items():
                train_config[k] = v
        else:
            for k, v in overrides.items():
                inference_config[k] = v
    
    setup_seeds(42)
    
    if args.mode == "train":
        train_model(train_config)
    else:

        inference_config["output_path"] = "predictions.csv"
        run_inference(inference_config)

if __name__ == "__main__":
    main()