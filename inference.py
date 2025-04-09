import torch
import numpy as np
from model import Wav2Vec2GrammarScoring
from dataset import prepare_test_dataloader

class Inference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.scores = []
        

        self.model = Wav2Vec2GrammarScoring(config["pretrained_model_name"]).to(
            self.device
        )


        self.model.load_state_dict(
            torch.load(config["model_path"], map_location=self.device)
        )
        self.model.eval()

        self.sampling_rate = config.get("sampling_rate", 16000)
        print("Inference model loaded and ready!")

        self.test_loader = prepare_test_dataloader(config)

    def predict(self):
        for batch in self.test_loader:
            waveform = batch["raw_waveform"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                output = self.model(waveform, attention_mask)
                batch_scores = output.squeeze(1).cpu().numpy()

                self.scores.append(batch_scores)

        all_scores = np.concatenate(self.scores, axis=0)
        print(f"Prediction completed! Generated {len(all_scores)} scores.")
        return all_scores
    
    def save_predictions(self, output_path="predictions.csv"):
        scores = self.predict()
        import pandas as pd
        

        test_df = pd.read_csv(self.config["test_metadata_path"])
        

        results_df = pd.DataFrame({
            "filename": test_df["filename"],
            "predicted_score": scores
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        return results_df