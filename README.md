# joint-vae-inverse-design
This project implements Variational Autoencoders (VAE) and a Joint VAE model with property prediction for the inverse design of materials. 
## Project Structure

- **`Joint-VAE.py`**: Contains the definition of the Joint VAE model, which includes an additional property prediction network.
- **`VAE.py`**: Contains the definition of the basic VAE model for material composition reconstruction.
- **`Joint-VAE_train.py`**: Script for training the Joint VAE model.
- **`VAE_train.py`**: Script for training the basic VAE model.
- **`data_processing.py`**: Contains functions for data preprocessing, including element indexing, composition matrix creation, and feature extraction from the dataset.
- **`requirements.txt`**: Lists the Python packages required to run the project.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/joint-vae-inverse-design.git
cd joint-vae-inverse-design
pip install -r requirements.txt
