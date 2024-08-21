import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from models import VAE
from data_processing import get_based_element_data, get_based_element_feature_data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    matrix, target_matrix, periodic_element_table = get_based_element_data(Based_element)
    feature_matrix = get_based_element_feature_data(composition_matrix, periodic_element_table)

    vae = VAE(in_features, latent_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr)

    num_epochs = 1000 
    Epoch_list = []
    loss_train = []
    loss_test = []

    for epoch in range(num_epochs):
        Epoch_list.append(epoch)
        total_loss = []
        total_recon = []
        total_kl = []

        vae.train()
        for x, _ in train_data_loader:
            x = x.to(device)
            x_reconst, mu, log_var = vae(x)

            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_recon.append(reconst_loss.item())
            total_kl.append(kl_div.item())

        avg_loss = sum(total_loss) / len(total_loss)
        avg_recon = sum(total_recon) / len(total_recon)
        avg_kl = sum(total_kl) / len(total_kl)
        loss_train.append(avg_recon)

        vae.eval()
        with torch.no_grad():
            total_test_loss = []
            total_test_recon = []
            total_test_kl = []

            for x, _ in test_data_loader:
                x = x.to(device)
                x_reconst, mu, log_var = vae(x)

                reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = reconst_loss + kl_div 

                total_test_loss.append(loss.item())
                total_test_recon.append(reconst_loss.item())
                total_test_kl.append(kl_div.item())

            avg_test_loss = sum(total_test_loss) / len(total_test_loss)
            avg_test_recon = sum(total_test_recon) / len(total_test_recon)
            avg_test_kl = sum(total_test_kl) / len(total_test_kl)
            loss_test.append(avg_recon)

        print(f'[Epoch {epoch + 1:03}/{num_epochs:03}] '
              f'loss: {avg_loss:.6f} Recon_loss: {avg_recon:.6f}, kl_loss: {avg_kl:.6f} sec')

    torch.save(vae.state_dict(), '')

