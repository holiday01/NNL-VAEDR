import torch
from model import AutoencoderClassifier
from train import train_and_evaluate
from Load import load_data
import argparse
from test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cancer type and distribution method.")
    parser.add_argument(
        "--cancer_type",
        type=str,
        choices=["breast", "cervical", "uterine"],
        default="cervical",
        help="Specify the cancer type: breast, cervical, or uterine."
    )

    parser.add_argument(
        "--reparam_method",
        type=str,
        choices=["normal", "gamma", "log_normal", "uniform", "log_gamma"],
        default="log_gamma",
        help="Specify the reparameterization method: normal, gamma, log_normal, uniform, or log_gamma."
    )
    parser.add_argument(
    "--is_test_run",
    action="store_true",
    default=False,
    help="Specify if this is a test run. Default is False."
    )
    args = parser.parse_args()
    cancer_type = args.cancer_type
    reparam_method = args.reparam_method
    is_test_run = args.is_test_run
    num = 512
    data, label, meta, exp = load_data(cancer_type, num)
    input_dim = data.shape[1]
    latent_dim = 32
    num_outputs = len(label[0])
    layers = [256, 128, latent_dim*2]
    class_layers = [8, 1]
    

    if is_test_run:
        test(input_dim, latent_dim, num_outputs, layers, class_layers, exp, reparam_method, cancer_type)
    else:
        train_and_evaluate(input_dim, latent_dim, num_outputs, layers, class_layers, data, label, meta, exp, reparam_method, cancer_type)
