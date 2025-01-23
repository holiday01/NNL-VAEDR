import matplotlib.pyplot as plt

def result_plot(train_metirc, vali_metric, reparam_method, fold, cancer_type):
    for key in ["accuracy", "recon_losses", "class_loss","f1", "auc"]:
        plt.figure(figsize=(8, 6))
        plt.plot(train_metirc[key], label=f"Train {key.capitalize()}", marker='o')
        plt.plot(vali_metric[key], label=f"Validation {key.capitalize()}", marker='x')
        plt.title(f"{key.capitalize()} Comparison: Train vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.legend()
        file_name = f"./Figure/{cancer_type}_{reparam_method}_Fold{fold}_{key}.pdf"
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()