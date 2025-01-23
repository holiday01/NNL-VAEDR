import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seedpy import SeedSetter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from Loss_F import auc, batch_sample_ccc
import Loss_F
from result_plot import result_plot
from test import test
from torch.utils.data import DataLoader, TensorDataset
from model import AutoencoderClassifier

# Create an instance with a specific seed
seed_setter = SeedSetter(seed=42)

# Set the seed
seed_setter.set_seed()


def train_and_evaluate(input_dim, latent_dim, num_outputs, layers, class_layers, data, label, meta, exp, reparam_method, cancer_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = Loss_F.get_loss_function(reparam_method)
    train_best, val_best, test_result = [],[],[]
    ae_lr, cl_lr = 0.0001, 0.0001
    for f in range(5):
        model = AutoencoderClassifier(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_outputs=num_outputs,
            encoder_layers= layers,
            decoder_layers= layers[::-1]+[input_dim],
            classifier_layers=class_layers,
            reparam_method=reparam_method
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        keys = ["accuracy", "recon_losses", "class_loss", "f1", "auc", "ccc",]
        metric = {mode: {key: [] for key in keys} for mode in ["train", "validation"]}
        
        # Scale features and labels
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data.values)  # Feature matrix
        y = np.array(label, dtype=float)  # Labels with possible NaN
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = X[meta.fold!=f], X[meta.fold==f], y[meta.fold!=f], y[meta.fold==f]
        
        # Convert to PyTorch tensors
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)

        # Define separate optimizers
        optimizer_autoencoder = optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=ae_lr,
            weight_decay=1e-5
        )
        optimizer_classifier = optim.Adam(
            list(model.classifiers.parameters()),
            lr=cl_lr,
            weight_decay=1e-5
        )
        
        
        num_epochs = 1000
        batch_size = 512
        best_val_class_loss = float('inf')  # Initialize to a very large value
        early_stop_patience = 30  # Number of epochs to wait for improvement
        no_improve_epochs = 0  # Counter for epochs without improvement
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        # Training and validation loop
        for epoch in range(num_epochs):
            # ===== Training Phase =====
            model.train()
            train_recon_loss = train_class_loss = train_correct = train_total = total_f1 = train_auc = train_batches = train_ccc = 0
            thre = 0
        
            for batch_X, batch_y in train_loader:
                # Mask NaN labels
                mask = ~torch.isnan(batch_y)
                if not mask.any():
                    continue
                masked_y = batch_y[mask]
        
                # === Step 1: Optimize Autoencoder ===
                optimizer_autoencoder.zero_grad()
                recon_X, pred_labels, mu, logvar = model(batch_X)
                recon_loss = loss_function(recon_X, batch_X, mu, logvar)
                recon_loss.backward(retain_graph=True)
                optimizer_autoencoder.step()
        
                # === Step 2: Optimize Classifier ===
                optimizer_classifier.zero_grad()
                recon_X, pred_labels, mu, logvar = model(batch_X)
                mask = ~torch.isnan(batch_y)
                if mask.any():
                    masked_pred = pred_labels[mask]
                    masked_y = batch_y[mask]
                    class_weights = torch.tensor([sum(masked_y == 0) / sum(masked_y == 1)], dtype=torch.float32).to(device)
                    classification_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)(masked_pred, masked_y)
                    classification_loss.backward()
                    optimizer_classifier.step()
        
                # Accumulate metrics
                train_recon_loss += recon_loss.item()
                train_class_loss += classification_loss.item()
                pred_binary = (masked_pred > thre).float()
                train_correct += (pred_binary == masked_y).sum().item()
                train_auc += auc(masked_y, nn.Sigmoid()(masked_pred))
                train_total += masked_y.numel()
                train_ccc += batch_sample_ccc(batch_X, recon_X)
        
                # Compute batch F1 score
                batch_f1 = f1_score(masked_y.cpu().numpy(), pred_binary.cpu().numpy(), zero_division=0, average='weighted')
                total_f1 += batch_f1 * masked_y.numel()
                train_batches += 1 
        
            # Final metrics for this epoch
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_recon_loss = train_recon_loss / train_batches if train_batches > 0 else 0
            train_class_loss = train_class_loss / train_batches if train_batches > 0 else 0
            train_f1 = total_f1 / train_total if train_total > 0 else 0
            train_auc = train_auc / train_batches if train_batches > 0 else  0
            train_ccc = train_ccc / train_batches if train_batches > 0 else  0
        
            metric["train"]["accuracy"].append(train_accuracy)
            metric["train"]["recon_losses"].append(train_recon_loss)
            metric["train"]["class_loss"].append(train_class_loss)
            metric["train"]["f1"].append(train_f1)
            metric["train"]["auc"].append(train_auc)
            metric["train"]["ccc"].append(train_ccc)
        
            # ===== Validation Phase =====
            model.eval()
            val_recon_loss = val_class_loss = val_correct = val_total = total_f1 = val_auc = valid_batches = val_ccc = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
            
                    # Mask NaN labels
                    mask = ~torch.isnan(batch_y)
                    if not mask.any():  # 如果 batch 全是 NaN，跳過
                        continue
                    masked_y = batch_y[mask]
            
                    # Forward pass
                    recon_X, pred_labels, mu, logvar = model(batch_X)
                    recon_loss = loss_function(recon_X, batch_X, mu, logvar)
                    val_recon_loss += recon_loss.item()
            
                    if mask.any():
                        masked_pred = pred_labels[mask]
                        class_weights = torch.tensor([sum(masked_y == 0) / sum(masked_y == 1)], dtype=torch.float32).to(device)
                        classification_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)(masked_pred, masked_y)
                        val_class_loss += classification_loss.item()
            
                        # Accumulate validation accuracy
                        pred_binary = (masked_pred > thre).float()
                        val_correct += (pred_binary == masked_y).sum().item()
                        val_total += masked_y.numel()
                        val_auc += auc(masked_y, nn.Sigmoid()(masked_pred))
                        val_ccc += batch_sample_ccc(batch_X, recon_X)
            
                        # Compute F1 score for the current batch
                        batch_f1 = f1_score(
                            masked_y.cpu().numpy(), 
                            pred_binary.cpu().numpy(), 
                            zero_division=0,
                            average='weighted'
                        )
                        total_f1 += batch_f1
                        valid_batches += 1  # 計算有效 batch
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            val_recon_loss = val_recon_loss / valid_batches
            val_class_loss = val_class_loss / valid_batches
            f1 = total_f1 / valid_batches if valid_batches > 0 else 0
            val_auc = val_auc / valid_batches if valid_batches > 0 else 0
            val_ccc = val_ccc / valid_batches if valid_batches > 0 else 0
            metric["validation"]["accuracy"].append(val_accuracy)
            metric["validation"]["recon_losses"].append(val_recon_loss)
            metric["validation"]["class_loss"].append(val_class_loss)
            metric["validation"]["f1"].append(f1)
            metric["validation"]["auc"].append(val_auc)
            metric["train"]["ccc"].append(val_ccc)
        
            # ===== Log Results =====
            print(f"Epoch {epoch + 1}/{num_epochs} \n"
                  f"Train Recon Loss: {train_recon_loss:.4f}, Train Class Loss: {train_class_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f} Train AUC: {train_auc:.4f} Train CCC: {train_ccc:.4f} \n "
                  f"Val Recon Loss: {val_recon_loss:.4f}, Val Class Loss: {val_class_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {f1:.4f} Val AUC: {val_auc:.4f} Val CCC: {val_ccc:.4f}")
        
            # ===== Early Stopping =====
            if val_class_loss < best_val_class_loss:
                best_val_class_loss = val_class_loss
                no_improve_epochs = 0
                train_m = [f, train_accuracy, train_auc, train_f1]
                val_m = [f, val_accuracy, val_auc, f1]
                torch.save(model.state_dict(), f"./checkpoint/{cancer_type}_{reparam_method}_best_model.pth")
                print(f"Epoch {epoch + 1}: Best model updated with val_class_loss={val_class_loss:.4f}")
            else:
                no_improve_epochs += 1
                print(f"Epoch {epoch + 1}: No improvement, val_class_loss={val_class_loss:.4f}")
                
            if no_improve_epochs >= early_stop_patience:
                train_best.append(train_m)
                val_best.append(val_m)
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        result_plot(metric["train"], metric["validation"], reparam_method, f, cancer_type)
        ac, au, sensitivity, specificity, f1, precision = test(input_dim, latent_dim, num_outputs, layers, class_layers, exp, reparam_method, cancer_type)
        test_result.append([f, ac, au, sensitivity, specificity, f1, precision])
    pd.DataFrame(test_result, columns = ["fold", "acc", "auc", "sensitivity", "specificity", "f1" , "precision"]).to_csv(f"./Acc/{cancer_type}_{reparam_method}_test_tresult.csv")
    pd.DataFrame(train_best, columns=["fold", "acc", "auc", "f1"]).to_csv(f"./Acc/{cancer_type}_{reparam_method}_train_tresult.csv")
    pd.DataFrame(val_best, columns=["fold", "acc", "auc", "f1"]).to_csv(f"./Acc/{cancer_type}_{reparam_method}_val_tresult.csv")
        
    
