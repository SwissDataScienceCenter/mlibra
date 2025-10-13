# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
experiment_dir = "./LGPALL-5-500-100010_log/"
experiment_dir = Path(experiment_dir)
import sys
sys.path.append("/myhome/mlibra/maldi/")
# !pip install -e /myhome/mlibra
from experiment import MaldiExperiment
from config import MaldiConfig
from utils import get_inducing_points
import numpy as np
args =np.load(experiment_dir / "args.npy", allow_pickle=True)
args = args.item()
from lgp_experiment import setup_experiment
experiment = setup_experiment(args)
experiment.load_train_sections()
experiment.load_train_coordinates()
experiment.load_train_samples()
experiment.load_train_data()

# %%
from l3di.simple_ddpm1d import SimpleUNet1D, SimpleDDPM1D
from torch.utils.data import TensorDataset

unet = SimpleUNet1D(in_channels=1, model_channels=64, out_channels=1, z_dim=5)
import torch
# Create DDPM model
ddpm = SimpleDDPM1D(unet, T=1000,var_type="fixedsmall")

experiment.lgp_model.load_state_dict(torch.load(experiment.config.exp_path / "model.pth", map_location=experiment.config.device))
vae = experiment.lgp_model
true_values = torch.tensor( np.load(experiment.config.exp_path /"train"/ "true_values.npy"))
coord = torch.tensor(experiment.coordinates_train)

diffusion_dataset = TensorDataset((true_values -true_values.min() )/ (true_values.max() - true_values.min()),coord)
dataloader_train= torch.utils.data.DataLoader(
    diffusion_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=0
)


# %%
ddpm.train(dataloader_train, epochs=2, lr=2e-4, vae=vae)

# %%
ddpm.train(dataloader_train, epochs=2, lr=1e-4, vae=vae)

# %%
ddpm.train(dataloader_train, epochs=2, lr=2e-4, vae=vae)

# %%
ddpm.train(dataloader_train, epochs=2, lr=1e-4, vae=vae)

# %%
ddpm.save_model(experiment.config.exp_path / "voxel_diffusion.pt")


# %%
ddpm.load_model(experiment.config.exp_path /"voxel_diffusion.pt", experiment.config.device)
x_t = torch.randn(1, 1, 176, device=experiment.config.device)
x_t

# %%
ddpm.var_type

# %%
import torch.nn.functional as F
padding = (0, 3) 

x = experiment.dataset_train[0][0]
coord = experiment.dataset_train[0][1]
coord = coord.unsqueeze(0)
coord = coord.to(experiment.config.device)
mu, logvar = vae.encode(coord)
z= vae.reparametrize(mu, logvar)
cond = vae.decode(z)
x_t = x_t.to(experiment.config.device)
cond = cond.to(experiment.config.device)
cond = F.pad(cond, padding, "constant", 0)
z = z.to(experiment.config.device)
cond = cond.unsqueeze(1)

sample_dict = ddpm.sample(x_t,cond,z,n_steps=500,guidance_weight=0,clip_denoised=False)

# %%
sample_dict['500'].shape

# %%
x_0 = sample_dict['500'].squeeze()
x_0 = x_0.cpu()
x_0 = x_0 *(true_values.max() - true_values.min()) + true_values.min() 
x_0=x_0[:173]

# %%
import matplotlib.pyplot as plt
plt.scatter(x.cpu().detach().numpy(),x_0[:173].detach().numpy())

# %%

# %%
plt.scatter(x.cpu().detach().numpy(),cond[:,:,:173].squeeze().cpu().detach().numpy())

# %%
#x = x.cpu().detach().numpy()
y= x_0[:173]
np.corrcoef(x,y)

# %%
z_=cond[:,:,:173].squeeze().cpu().detach().numpy()
x=x.numpy()
y=y.numpy()
plt.scatter(x,cond[:,:,:173].squeeze().cpu().detach().numpy())
np.corrcoef(x,z_)

# %%
np.corrcoef(x,z_-y)

# %%
np.corrcoef(y,z_)

# %%
plt.scatter(z_,y)

# %%
plt.plot(y)

# %%
from scipy import stats
stats.spearmanr(x,y)

# %%
stats.spearmanr(x,z_-y)

# %%
stats.spearmanr(x,z_)

# %%
np.mean((x-y)**2)

# %%
np.mean((x-z_)**2)

# %%
np.mean((x-(z_-y))**2)

# %%
np.mean((x-y)/x)

# %%
np.mean((x-z_)/x)

# %%
np.mean((x-(z_-y))/x)

# %%
true_values_max = true_values.max()
true_values_min = true_values.min()

# %%
import logging
from tqdm import tqdm
if(experiment.config.exp_path / "model.pth").exists():
    logging.info("Loading model from file")
    experiment.lgp_model.load_state_dict(torch.load(experiment.config.exp_path / "model.pth", map_location=experiment.config.device))
    logging.info("Model loaded successfully")

volume_path = experiment.config.exp_path / "volume_diffusion"
volume_path.mkdir(parents=True, exist_ok=True)
template_file = volume_path / "template_volume.npy"
if not template_file.exists():
    logging.info("Downloading template volume...")
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    # Specify the resolution you want for the template volume (in microns)
    resolution_um = 25
    mcc = MouseConnectivityCache(resolution=resolution_um)
    logging.info(f"Downloading/loading reference TEMPLATE volume at {resolution_um} um resolution...")
    template_volume, _ = mcc.get_template_volume()
    logging.info(f"Template volume shape: {template_volume.shape}")
    logging.info(f"Template volume data type: {template_volume.dtype}")
    np.save(template_file, template_volume)
else:
    logging.info("Template volume already exists, loading from file")
    template_volume = np.load(template_file)
non_zero_indices = np.argwhere(template_volume > 5)
# concert from 25 um to 1 mm
non_zero_ccf = non_zero_indices * 0.025
non_zero_ccf = torch.tensor(non_zero_ccf, dtype=torch.float32)
non_zero_ccf = (non_zero_ccf - experiment.coord_mean) / experiment.coord_std
ccf_dataset = torch.utils.data.TensorDataset(torch.tensor(non_zero_ccf, dtype=torch.float32), torch.tensor(non_zero_indices, dtype=torch.float32))
ccf_dataloader = torch.utils.data.DataLoader(ccf_dataset, batch_size=experiment.config.batch_size, shuffle=False, num_workers=0)
logging.info("Reconstructing whole brain volume...")

experiment.lgp_model.eval()
with torch.no_grad():
    for i,batch in enumerate(tqdm(ccf_dataloader)):
        batch_file = volume_path / f"batch_{i}.pth"
        if batch_file.exists():
            logging.info(f"Batch {i} already exists, skipping...")
            continue
        coordinates_batch = batch[0].to(experiment.config.device)
        indices_batch = batch[1]
        cond, gp_posterior = experiment.lgp_model.predict(coordinates_batch)
        x_t = torch.randn(experiment.config.batch_size, 1, 176, device=experiment.config.device)
        cond = F.pad(cond, padding, "constant", 0)
        z = z.to(experiment.config.device)
        cond = cond.unsqueeze(1)

        sample_dict = ddpm.sample(x_t,cond,gp_posterior.mean,n_steps=500,guidance_weight=0,clip_denoised=False)
        cond = cond.squeeze()
        # we save the coordinates the indices and the predictions in a single file
        predictions = sample_dict['500'].squeeze().detach().cpu()
        predictions = (predictions)*(true_values_max - true_values_min) + true_values_min
        predictions = predictions[:,:173].cpu()
        predictions = cond[:,:173].detach().cpu() - predictions
        predictions[:,:173] = predictions[:,:173]*experiment.train_std.cpu() + experiment.train_mean.cpu()
        predictions = predictions.numpy()
        if experiment.config.log_transform:
           predictions = np.exp(predictions) - 1e-10
        torch.save({
           "coordinates": coordinates_batch.cpu(),
           "indices": indices_batch.cpu(),
           "predictions": predictions,
        }, batch_file)



# %%
sample_dict['1000'].shape

# %%
predictions.shape

# %%
cond.shape

# %%
from tqdm import tqdm
import torch
import logging
def load_whole_brain_reconstruction(experiment, lipid):
    """
    Loads the whole brain reconstruction for a specific lipid.
    """
    volume_path = experiment.config.exp_path / "volume_diffusion"
    template_file = volume_path / "template_volume.npy"
    if not template_file.exists():
        logging.error("Template volume does not exist. Please run whole_brain_reconstruction() first.")
        return None
    template_volume = np.load(template_file)
    non_zero_indices = np.argwhere(template_volume > 5)
    # fill template volume with zeros
    template_volume = np.zeros_like(template_volume, dtype=np.float32)

    lipid_name = experiment.config.selected_lipids_names[lipid]
    lipid_volume_name = experiment.config.exp_path / f"{lipid_name}_volume_diffusion.npy"
    if lipid_volume_name.exists():
        logging.info(f"Lipid volume for {lipid_name} already exists, loading from file")
        lipid_volume = np.load(lipid_volume_name)
        return lipid_volume
    else:
        for i in tqdm(range(len(non_zero_indices) // experiment.config.batch_size + 1)):
            batch_file = volume_path / f"batch_{i}.pth"
            if not batch_file.exists():
                continue
            batch_data = torch.load(batch_file)
            template_volume[batch_data["indices"][:, 0].long(),
                            batch_data["indices"][:, 1].long(),
                            batch_data["indices"][:, 2].long()] = batch_data["predictions"][:, lipid]
        # Save the reconstructed lipid volume
        # We need to ensure the volume is saved in the same shape as the template
        # set 0 to nan
        template_volume[template_volume == 0] = np.nan

        np.save(lipid_volume_name, template_volume)
        logging.info(f"Lipid volume for {lipid_name} saved to {lipid_volume_name}")
        template_volume = 255*( template_volume - np.nanmin(template_volume)) / (np.nanmax(template_volume)- np.nanmin(template_volume))
        np.save(experiment.config.exp_path / f"{lipid_name}_volume255_diffusion.npy", template_volume)
        return template_volume

load_whole_brain_reconstruction(experiment,33)

# %%
temp = np.load("/myhome/data/maldi/lmmvae/LGPALL-5-500-100010_log/HexCer 42:2;O2_volume_diffusion.npy")

# %%
temp = 255 *(temp - np.nanmin(temp)) / (np.nanmax(temp) - np.nanmin(temp))

# %%
np.save("/myhome/data/maldi/lmmvae/LGPALL-5-500-100010_log/HexCer 42:2;O2_volume255_diffusion.npy", temp)

# %%
