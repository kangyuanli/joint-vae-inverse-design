import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from models import Joint_VAE
from data_processing import get_based_element_data, get_based_element_feature_data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    matrix, target_matrix, periodic_element_table = get_based_element_data()
    feature_matrix = get_based_element_feature_data(composition_matrix=matrix, periodic_element_table=periodic_element_table)

    matrix_tensor = torch.from_numpy(matrix).float()
    target_tensor = torch.from_numpy(target_matrix).float()



    vae = Joint_VAE(in_features, latent_size).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr)

    num_epochs = 
    Epoch_list = []
    loss_train = []
    loss_test = []

    for epoch in range(num_epochs):
        Epoch_list.append(epoch)
        total_loss = []
        total_recon = []
        total_kl = []
        total_pro = []

        vae.train()
        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)

            x_reconst, mu, log_var, Properties = vae(x)

            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
            properties_loss = F.mse_loss(Properties, y, reduction='mean')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div + properties_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_recon.append(reconst_loss.item())
            total_kl.append(kl_div.item())
            total_pro.append(properties_loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        avg_recon = sum(total_recon) / len(total_recon)
        avg_kl = sum(total_kl) / len(total_kl)
        avg_pro = sum(total_pro) / len(total_pro)
        loss_train.append(avg_recon)

        print(f'[Epoch {epoch + 1:03}/{num_epochs:03}] '
              f'loss: {avg_loss:.6f} Recon_loss: {avg_recon:.6f}, kl_loss: {avg_kl:.6f}, pro_loss: {avg_pro:.6f}')

        vae.eval()
        with torch.no_grad():
            total_test_loss = []
            total_test_recon = []
            total_test_kl = []
            total_test_pro = []

            for x, y in test_data_loader:
                x = x.to(device)
                y = y.to(device)

                x_reconst, mu, log_var, Properties = vae(x)

                reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
                properties_loss = F.mse_loss(Properties, y, reduction='mean')
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = reconst_loss + kl_div * 1e-4 + properties_loss * 1e-1

                total_test_loss.append(loss.item())
                total_test_recon.append(reconst_loss.item())
                total_test_kl.append(kl_div.item())
                total_test_pro.append(properties_loss.item())

            avg_test_loss = sum(total_test_loss) / len(total_test_loss)
            avg_test_recon = sum(total_test_recon) / len(total_test_recon)
            avg_test_kl = sum(total_test_kl) / len(total_test_kl)
            avg_test_pro = sum(total_test_pro) / len(total_test_pro)
            loss_test.append(avg_recon)

            print(f'[Epoch {epoch + 1:03}/{num_epochs:03}] '
                  f'loss: {avg_test_loss:.6f} Recon_test_loss: {avg_test_recon:.6f}, '
                  f'kl_test_loss: {avg_test_kl:.6f}, pro_test_loss: {avg_test_pro:.6f}')

    torch.save(vae.state_dict(), '')



