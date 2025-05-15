from ClipBertViewPredictor import ClipBertViewPredictor
from DatasetCreator import Dataset
from YouTubeDataset import YouTubeDataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

def custom_msle_loss(y_true, y_pred): # to change after for efficiency
    y_true = torch.exp(y_true) 
    y_pred = torch.exp(y_pred)
    return torch.mean((torch.log1p(y_true) - torch.log1p(y_pred))**2)

def train_model(model, train_dataset, val_dataset, epochs=5, lr=1e-3, batch_size=2, optimizer=None, scheduler=None, early_stopping=False, patience=5):

    training_losses = []
    validation_losses = []

    print("Training model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])
        }

    print("Creating DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    criterion = custom_msle_loss
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0 # Early stopping counter

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True, pin_memory=True)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # Why ?
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation

        print("Validating model...")
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in progress_bar:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != "target"}
                targets = batch["target"].to(device)

                outputs = model(inputs).view(-1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
            
                progress_bar.set_postfix(loss=loss.item())
            
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                print(f"Validation loss improved to {avg_val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), f'./models/best_model_clipbertpredictor.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}/{epochs}")
                    break

        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
    
    return training_losses, validation_losses

def predict_and_save(model, test_dataset, batch_size=8):

    
    print("Testing model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "id": [x["id"] for x in batch]
        }
    predictions=[]
    ids=[]
    print("Creating DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)    
    progress_bar = tqdm(test_loader,desc="Testing", leave=False)
    for batch in progress_bar:
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items() if k not in ["target","id"]}
        outputs = model(inputs).view(-1).cpu()  # Move to CPU immediately
        predictions.extend(outputs.detach().numpy())  # Store as numpy array
        ids.extend(batch["id"])
    df = pd.DataFrame({
        'ID': ids,  # Or any identifier column
        'views': predictions  # Match competition requirements
    })
    df["views"]=np.exp(df["views"])
    assert len(df) == 3402, f"Expected 3402 rows, got {len(df)}"
    
    df.to_csv("./artifacts/results.csv", index=False)
    print(f"Submission saved")


if __name__ == '__main__':
    # Train and test
    ratio = 1
    training_data = YouTubeDataset(ratio=ratio, csv_path="./dataset/processed_training_set.csv", root_dir="./dataset/train_val/")
    val_data = YouTubeDataset("./dataset/processed_validation_set.csv", "./dataset/train_val/", ratio=ratio)
    test_data = YouTubeDataset("./dataset/processed_test_set.csv", "./dataset/test/", training = False)
    model = ClipBertViewPredictor()

    epochs = 100
    lr = 1e-3
    batch_size = 16
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = True
    patience = 5

    train_losses, val_losses = train_model(model, training_data, val_data, optimizer=optimizer, scheduler=scheduler, epochs=epochs, batch_size=batch_size, early_stopping=early_stopping, patience=patience)
    predict_and_save(model, test_data)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss (MSLE)')
    plt.legend()
    plt.savefig('./artifacts/clipbert_predictor_loss_plot.png')