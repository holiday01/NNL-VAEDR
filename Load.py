import pandas as pd

def load_data(cancer_type, num):
    meta = pd.read_csv("../Data/20_meta.csv", index_col=0)
    data = pd.read_csv("../Data/20_sgdc.csv")
    label = pd.read_csv("../Data/20_label.csv").values.tolist()
    if cancer_type == "breast":
        exp = pd.read_csv("../../../brac/brac_exp.csv")
    elif cancer_type == "cervical":
        exp = pd.read_csv("../../../brac/cervical_exp.csv")
    elif cancer_type == "uterine":
        exp = pd.read_csv("../../../brac/uterine_exp.csv")
    gene = data.columns
    intersection = list(set(exp.columns[1:].values) & set(gene))
    data = data[intersection]
    median_var = data.var().median()
    top_gene = (data.var() - median_var).abs().nlargest(num).index.tolist()
    data = data[top_gene]
    exp = exp[top_gene]

    return data, label, meta, exp