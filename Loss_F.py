import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score

beta = 0.001

def auc(x,y):
        fpr, tpr, thresholds = metrics.roc_curve(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        return metrics.auc(fpr, tpr)

def batch_sample_ccc(batch_X, recon_X):
    ccc_list = []
    for x, recon in zip(batch_X, recon_X):
        x_mean, recon_mean = x.mean(), recon.mean()
        x_var, recon_var = x.var(), recon.var()
        covariance = ((x - x_mean) * (recon - recon_mean)).mean()
        numerator = 2 * covariance
        denominator = x_var + recon_var + (x_mean - recon_mean) ** 2
        ccc = numerator / (denominator + 1e-6)  # Add small value to avoid division by zero
        ccc_list.append(ccc.item())
    return torch.tensor(ccc_list).mean().item()



def kl_divergence_loggamma(log_k, log_theta, prior_k=1.0, prior_theta=1.0):
    """Compute KL divergence between Gamma(q) and Gamma(p)."""
    k_q = torch.clamp(torch.exp(log_k), min=1e-5, max=5)
    theta_q = torch.clamp(torch.exp(log_theta), min=1e-5, max=5)
    k_p = torch.tensor(prior_k, device=log_k.device)
    theta_p = torch.tensor(prior_theta, device=log_theta.device)
    psi_k_q = torch.digamma(k_q)
    kl = (
        k_q * torch.log(theta_q / theta_p)
        - torch.lgamma(k_q) + torch.lgamma(k_p)
        + (k_q - k_p) * psi_k_q
        + k_p * (theta_q - theta_p) / theta_p
    )
    return kl.mean()

def loss_log_gamma(reconstruction, x, log_k, log_theta, beta = beta):
    """Loss function for Gamma distribution."""
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
    kl_div = kl_divergence_loggamma(log_k, log_theta)
    return recon_loss + beta * kl_div

def kl_divergence_gamma(log_k, log_theta, prior_k=1.0, prior_theta=1.0):
    """Compute KL divergence between Gamma(q) and Gamma(p)."""
    k_q = torch.clamp(log_k, min=1e-5, max=5)
    theta_q = torch.clamp(log_theta, min=1e-5, max=5)
    k_p = torch.tensor(prior_k, device=log_k.device)
    theta_p = torch.tensor(prior_theta, device=log_theta.device)
    psi_k_q = torch.digamma(k_q)
    kl = (
        k_q * torch.log(theta_q / theta_p)
        - torch.lgamma(k_q) + torch.lgamma(k_p)
        + (k_q - k_p) * psi_k_q
        + k_p * (theta_q - theta_p) / theta_p
    )
    return kl.mean()

def loss_gamma(reconstruction, x, log_k, log_theta, beta = beta):
    """Loss function for Gamma distribution."""
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
    kl_div = kl_divergence_gamma(log_k, log_theta) 
    return recon_loss + beta * kl_div

def loss_gaussian(reconstruction, x, mu, logvar, beta = beta):
    """Loss function for Gaussian distribution."""
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  / logvar.numel()
    return recon_loss + beta * kl_div

def loss_log_normal(reconstruction, x, mu, log_var, beta = beta):
    """Loss function for Log-Normal distribution."""
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / log_var.numel()
    return recon_loss + beta * kl_div

def loss_uniform(reconstruction, x, mu, log_var, beta = beta):
    """Loss function for Uniform distribution."""
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(log_var)
    return recon_loss + beta * kl_div

def get_loss_function(loss_type):
    """Return the appropriate loss function based on loss_type."""
    if loss_type == "log_gamma":
        return loss_log_gamma
    elif loss_type== "gamma":
        return loss_gamma
    elif loss_type == "normal":
        return loss_gaussian
    elif loss_type == "log_normal":
        return loss_log_normal
    elif loss_type == "uniform":
        return loss_uniform
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")