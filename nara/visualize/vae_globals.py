import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle, os
import numpy as np

class VAE_2L(nn.Module):
    def __init__(self, vae_config):
        super().__init__()

        self.vae_config = vae_config

        input_dim = vae_config["input_dim"]
        hidden_dim = vae_config["hidden_dim"]
        latent_dim = vae_config["latent_dim"]

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim) # fc_mu
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim) # fc_logvar

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar, inference=False):
        if inference:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def train_epoch(self, dataloader, optimizer, device, decoder_only=False):
        self.train() # training mode!
        total_loss = 0
        for data, in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            if decoder_only:
                self.encoder.eval()
                self.mean_layer.eval()
                self.decoder.train()
                with torch.no_grad():
                    x_recon = self.encoder(data)
                    x_recon = self.mean_layer(x_recon)
                x_recon = self.decoder(x_recon)
                loss = loss_function_recon(x_recon, data)
            else:
                x_recon, mu, logvar = self.forward(data)
                loss = loss_function_ELBO(x_recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    @torch.no_grad
    def validate_epoch(self, dataloader, device, decoder_only=False):
        self.eval() # evaluation mode!
        total_loss = 0
        for data, in dataloader:
            data = data.to(device)
            if decoder_only:
                x_recon = self.encoder(data)
                x_recon = self.mean_layer(x_recon)
                x_recon = self.decoder(x_recon)
                loss = loss_function_recon(x_recon, data)
            else:
                x_recon, mu, logvar = self(data)
                loss = loss_function_ELBO(x_recon, data, mu, logvar)
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    def save(self, filename:str = "vae_best_model.pt"):
        save_dict = dict(
            model_state_dict = self.state_dict(),
            vae_config = self.vae_config,
            )
        torch.save(save_dict, filename)

    @classmethod
    def load(cls, filename:str, device:str = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        loaded_data = torch.load(filename, map_location = device)
        vae_config = loaded_data["vae_config"]
        loaded_model = cls(vae_config = vae_config)
        loaded_model.load_state_dict(loaded_data["model_state_dict"])
        loaded_model.to(device=device)
        return loaded_model

def loss_function_recon(x_recon, x):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    return recon_loss

def loss_function_ELBO(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + KL_divergence  # Total loss = reconstruction loss + KL divergence
    return loss


def main(device="cpu", dtype="float64"):

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            device = device
        else:
            raise IOError("device arguments should be either str or torch.device instance")

    dt = torch.float32 if dtype.lower()=="float32" else torch.float64

    hidden_dim = 16
    latent_dim = 2

    max_num_epochs = 1000
    patience = 30
    train_vali_ratio = 0.8 # Train: 80% / Validation: 20%

    # input data
    BH_path = "/home2/giyeok/project/1_ing/3_NARA/Applications/4_8_structure/1_BH"
    X_globals = []
    for i in range(34):
        fn = f"{i+1:0>2d}"
        with open(os.path.join(BH_path, fn, "ab.pickle"), 'rb') as fi:
            _x_ab = pickle.load(fi)
        for _x in _x_ab:
            X_globals.append(torch.sum(_x, dim=0))

    # To tensor & Dataset separation: Training dataset & Validation dataset
    X_globals = torch.stack(X_globals, dim=0).to(dtype=dt)
    input_dim = X_globals.shape[1]
    dataset = TensorDataset(X_globals)

    train_size = int(train_vali_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VAE_2L(
        vae_config = dict(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            latent_dim = latent_dim
        ))
    if dtype=="float64":
        model.double()

    model = model.to(device)
    optimizer = optim.NAdam(model.parameters())

    best_val_loss = None
    early_stop = False
    decoder_only = False
    epochs_no_improve = 0

    for epoch in range(max_num_epochs):
        avg_train_loss = model.train_epoch(train_loader, optimizer, device, decoder_only=decoder_only)
        avg_val_loss = model.validate_epoch(val_loader, device, decoder_only=decoder_only)
        print(f'Epoch [{epoch+1}/{max_num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save('VAE_best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                early_stop = True
                break

            #### When if we need decoder
            # if decoder_only:
            #     print('Early stopping!')
            #     early_stop = True
            #     break
            # else:
            #     print("Training done. train decoder further")
            #     decoder_only = True
            #     epochs_no_improve = 0
            #     optimizer = optim.NAdam(model.decoder.parameters())                

#    model.load_state_dict(torch.load('VAE_best_model.pt', weights_only=True))
    model = model.load("VAE_best_model.pt", device=device)
    model.eval()

    all_data_loader = DataLoader(dataset, batch_size = 256, shuffle = False)
    all_coords_latent = []
    with torch.no_grad():
        for data, in all_data_loader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            all_coords_latent.append(mu.cpu().numpy())
    all_coords_latent = np.vstack(all_coords_latent)
    np.savetxt("All_coords_latent.txt", all_coords_latent)

if __name__=="__main__":
    if torch.cuda.is_available():
        _available_device = "cuda"
    elif torch.backends.mps.is_available():
        _available_device = "mps"
    else:
        _available_device = "cpu"
    # _available_device = "cuda" if torch.cuda.is_available() else "cpu"

    if _available_device.lower() == "cpu":
        num_cpus = len(os.sched_getaffinity(0))
        torch.set_num_threads(num_cpus)
        print(num_cpus, "cpus are used")
    else:
        print(_available_device, "is used")

    dtype = "float64"

    # torch.manual_seed(1234)
    # np.random.seed(1234)

    main(_available_device, dtype)
