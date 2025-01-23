import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from model import AutoencoderClassifier

def test(input_dim, latent_dim, num_outputs, layers, class_layers, exp, reparam_method, cancer_type):
    model = AutoencoderClassifier(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_outputs=num_outputs,
        encoder_layers= layers,
        decoder_layers= layers[::-1]+[input_dim],
        classifier_layers=class_layers,
        reparam_method=reparam_method
    )
    model.load_state_dict(torch.load(f"./checkpoint/{cancer_type}_{reparam_method}_best_model.pth"))
    model = model.to(torch.device('cuda'))
    model.eval()
    if cancer_type == "breast":
        drug_mata = pd.read_csv("../../../brac/tcga_brca_clinical.csv")
        cancer_id = pd.read_csv("../../../brac/brac_exp.csv")['Case ID']
        l = 5
    elif cancer_type == "cervical":
        drug_mata = pd.read_csv("../../../brac/tcga_cervical_clinical.csv")
        cancer_id = pd.read_csv("../../../brac/cervical_exp.csv")['Case ID']
        l = 170
    elif cancer_type == "uterine":
        drug_mata = pd.read_csv("../../../brac/tcga_uterine_clinical.csv")
        cancer_id = pd.read_csv("../../../brac/uterine_exp.csv")['Case ID']
        l = 5
    
    resp = drug_mata.replace({'Complete Response': 1, 'Partial Response': 1, 'Clinical Progressive Disease':0,'Stable Disease': 0})
    resp.index = resp['bcr_patient_barcode']
    resp = [resp[resp.index == n]['measure_of_response'].values[0] for n in cancer_id]
    scaler = MinMaxScaler()
    X_test_tensor = torch.tensor(scaler.fit_transform(exp.values), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        recon_X, pred_labels, mu, logvar = model(X_test_tensor.to(torch.device('cuda')))

    
    weights = compute_sample_weight(class_weight='balanced', y=resp)
    fpr, tpr, thresholds = metrics.roc_curve(resp, [n[l] for n in nn.Sigmoid()(pred_labels).cpu().detach().numpy()], sample_weight=weights)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    pred_binary = (nn.Sigmoid()(pred_labels) > optimal_threshold).float()
    pred = pd.DataFrame(pred_binary.cpu())[l].values
    pred_score = pd.DataFrame(nn.Sigmoid()(pred_labels).cpu())[l].values
    
    # 計算 Sensitivity 和 Specificity
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    #auc = metrics.auc(fpr, tpr)
    auc = metrics.roc_auc_score(resp, pred_score,average="micro")
    f1 = metrics.f1_score(resp, pred)
    precision = metrics.precision_score(resp, pred)
    
    # 顯示結果
    print("Testing \n")
    print("ACC:",sum(pred == resp) / len(resp))
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (1-FPR): {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1: {f1:.4f}")
    return sum(pred == resp) / len(resp), auc, sensitivity, specificity, f1, precision

