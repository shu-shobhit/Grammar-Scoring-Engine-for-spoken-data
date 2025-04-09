# trainer.py
import torch
import torch.nn as nn
import numpy as np
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from torchmetrics import MeanSquaredError
import wandb

from model import Wav2Vec2GrammarScoring
from dataset import prepare_dataloaders

class Trainer:
    def __init__(self, config, model=None):
        self.config = config
        self.device = torch.device(config["device"])
        

        self.train_loader, self.val_loader = prepare_dataloaders(config)
        

        self.criterion = nn.MSELoss()
        self.rmse_metric = MeanSquaredError(squared=False).to(self.device)
        

        if model:
            self.model = model.to(self.device)
        else:
            self.model = Wav2Vec2GrammarScoring(
                config["pretrained_model_name"],
            ).to(self.device)
            self.model.freeze_feature_encoder()
            print("Model initialized!")


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.95
        )
        

        if self.config["use_wandb"]:
            wandb.init(project=config["project_name"], config=config)
        
        self.history = {
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "val_rmse": [],
        }
        self.best_val_rmse = float("inf")

    def train(self):
        step = 0
        running_loss = 0.0

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            self.rmse_metric.reset()

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn("• Loss: {task.fields[loss]}", justify="right"),
                transient=True,
            ) as progress:
                train_task = progress.add_task(
                    f"Epoch {epoch + 1}/{self.config['num_epochs']} [bold white on blue]TRAIN[/]",
                    total=len(self.train_loader),
                    loss="0.0000"
                )
                for batch in self.train_loader:
                    waveforms = batch["raw_waveform"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device).squeeze(1)  # [B, 1]

                    self.optimizer.zero_grad()

                    outputs = self.model(waveforms, attention_mask)  # Shape: [B, 1]
                    loss = self.criterion(outputs.squeeze(1), labels)

                    loss.backward()
                    self.optimizer.step()
                    step += 1
                    running_loss += loss.item()
                    self.rmse_metric.update(outputs.squeeze(1), labels)

                    progress.update(
                        train_task,
                        advance=1,
                        loss=f"{(running_loss) / (step):.4f}"
                    )
                    
                    if self.config["use_wandb"]:
                        wandb.log({
                            "train_step_loss": (running_loss) / (step)
                        }, step=step)
                    
                    if step % 100 == 0:
                        train_loss = (running_loss) / (step)
                        train_rmse = self.rmse_metric.compute().item()
            
                        val_loss, val_rmse = self.evaluate(epoch, progress)
            
                        self.history["train_loss"].append(train_loss)
                        self.history["train_rmse"].append(train_rmse)
                        self.history["val_loss"].append(val_loss)
                        self.history["val_rmse"].append(val_rmse)
            
                        if self.config["use_wandb"]:
                            wandb.log(
                                {
                                    "epoch": epoch + 1,
                                    "train_rmse": train_rmse,
                                    "val_loss": val_loss,
                                    "val_rmse": val_rmse,
                                },
                                step=step,
                            )
            
                        if val_rmse < self.best_val_rmse:
                            self.best_val_rmse = val_rmse
                            model_path = "best_model.pt"
                            torch.save(self.model.state_dict(), model_path)
                            if self.config["use_wandb"]:
                                wandb.save(model_path)
                                self._log_model_as_artifact(model_path)
            
                        print(
                            f"[Epoch {epoch + 1}] [Step {step}] Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f} | Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}"
                        )

            self.scheduler.step()

        if self.config["use_wandb"]:
            wandb.finish()
        print("Training completed!")
        
        return self.history, self.model

    def evaluate(self, epoch, progress):
        self.model.eval()
        val_loss = 0.0
        self.rmse_metric.reset()

        with torch.no_grad():
            val_task = progress.add_task(
                f"Epoch {epoch + 1}/{self.config['num_epochs']} [bold white on green]VAL[/]", 
                total=len(self.val_loader), 
                loss="0.0000",
            )
            for i, batch in enumerate(self.val_loader):
                waveforms = batch["raw_waveform"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).squeeze(1)
    
                outputs = self.model(waveforms, attention_mask)
                loss = self.criterion(outputs.squeeze(1), labels)
                val_loss += loss.item()
                self.rmse_metric.update(outputs.squeeze(1), labels)
    
                progress.update(
                    val_task, advance=1, loss=f"{(val_loss) / (i+1):.4f}"
                )

        return val_loss / len(self.val_loader), self.rmse_metric.compute().item()

    def _log_model_as_artifact(self, model_path):
        if self.config["use_wandb"]:
            artifact = wandb.Artifact(f"best_model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)