import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform, Gamma

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, num_outputs, encoder_layers, decoder_layers, classifier_layers, reparam_method="normal"):
        super(AutoencoderClassifier, self).__init__()
        self.reparam_method = reparam_method
        # Build Encoder
        self.encoder = self._build_layers(input_dim, encoder_layers, use_batchnorm=True, dropout=0.6, activation='relu')
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

        # Build Decoder
        self.decoder = self._build_layers(latent_dim, decoder_layers[:-1], activation='relu')
        self.decoder.add_module('output', nn.Sequential(nn.Linear(decoder_layers[-2], decoder_layers[-1]), nn.Sigmoid()))

        # Build Classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                *self._build_layers(latent_dim, classifier_layers[:-1], activation='relu'),
                nn.Linear(classifier_layers[-2], classifier_layers[-1]),
            )
            for _ in range(num_outputs)
        ])

    def _build_layers(self, input_dim, layer_dims, use_batchnorm=False, dropout=0.0, activation='relu'):
        layers = []
        for dim in layer_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm:  # Batch Normalization
                layers.append(nn.BatchNorm1d(dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu': 
                layers.append(nn.LeakyReLU(0.2))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = dim
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        hidden_dim = h.size(1)
        half_dim = hidden_dim // 2 
        hm = h[:, :half_dim]
        mu = self.mu(hm)
        hl = h[:, half_dim:]
        logvar = self.logvar(hl)
        return mu, logvar, h

    def reparameterize(self, mu, logvar, reparam_method):
        if reparam_method == "normal":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        elif reparam_method == "log_normal":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return torch.exp(mu + eps * std)
        elif reparam_method == "uniform":
            lower = mu - torch.exp(0.5 * logvar)
            upper = mu + torch.exp(0.5 * logvar)
            return torch.distributions.Uniform(lower, upper).rsample()
        elif reparam_method == "gamma":
            alpha = mu.clamp(min=1e-5, max=5)
            beta = logvar.clamp(min=1e-5, max=5)
            return torch.distributions.Gamma(alpha, beta).rsample()
        elif reparam_method == "log_gamma":
            alpha = torch.exp(mu).clamp(min=1e-5, max=5)
            beta = torch.exp(logvar).clamp(min=1e-5, max=5)
            return torch.distributions.Gamma(alpha, beta).rsample()
        else:
            raise ValueError(f"Invalid reparam_method: {reparam_method}")

    def decode(self, z):
        return self.decoder(z)

    def classify(self, z):
        return torch.cat([classifier(z) for classifier in self.classifiers], dim=1)

    def forward(self, x):
        mu, logvar, h = self.encode(x)
        z = self.reparameterize(mu, logvar, self.reparam_method)
        recon_x = self.decode(z)
        latent_features = torch.cat((mu, logvar), dim=1)
        pred_labels = self.classify(z)
        return recon_x, pred_labels, mu, logvar
