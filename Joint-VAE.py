import torch
import torch.nn as nn
import torch.nn.functional as F

class Joint_VAE(nn.Module):

    def __init__(self, in_features, latent_size):
        super(VAE, self).__init__()

        self.encoder_fcl_1 = nn.Linear(in_features, 90)
        self.encoder_fcl_1_norm = nn.LayerNorm(90)
        self.encoder_fcl_2 = nn.Linear(90, 48)
        self.encoder_fcl_2_norm = nn.LayerNorm(48)
        
        self.latent_mu = nn.Linear(48, latent_size)
        self.latent_var = nn.Linear(48, latent_size)

        self.decoder_fcl_1 = nn.Linear(latent_size, 48)
        self.decoder_fcl_1_norm = nn.LayerNorm(48)
        self.decoder_fcl_2 = nn.Linear(48, 90)
        self.decoder_fcl_2_norm = nn.LayerNorm(90)
        self.out_fcl = nn.Linear(90, in_features)

        self.predict_fcl_1 = nn.Linear(latent_size, 60)
        self.predict_fcl_1_norm = nn.LayerNorm(60)
        self.predict_fcl_2 = nn.Linear(60, 30)
        self.predict_fcl_2_norm = nn.LayerNorm(30)
        self.predict_out = nn.Linear(30, 1)

    def encoder(self, X):
        out = self.encoder_fcl_1(X)
        out = F.relu(self.encoder_fcl_1_norm(out))
        out = self.encoder_fcl_2(out)
        out = F.relu(self.encoder_fcl_2_norm(out))

        mu = self.latent_mu(out)
        log_var = self.latent_var(out)

        return mu, log_var

    def decoder(self, z):
        h = self.decoder_fcl_1(z)
        h = F.relu(self.decoder_fcl_1_norm(h))
        h = self.decoder_fcl_2(h)
        h = F.relu(self.decoder_fcl_2_norm(h))
        x_reconst = F.softmax(self.out_fcl(h), dim=1)
        return x_reconst
    
    def predict(self, z):
        p = self.predict_fcl_1(z)
        p = F.relu(self.predict_fcl_1_norm(p))
        p = self.predict_fcl_2(p)
        p = F.relu(self.predict_fcl_2_norm(p))
        pro_out = self.predict_out(p)
        return pro_out

    def reparameterization(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        x_reconst = self.decoder(z)
        pro_out = self.predict(z)
        return x_reconst, mu, log_var, pro_out