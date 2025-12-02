import torch
import argparse
import yaml
import os
from tqdm import tqdm
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from dataset.dataset import CtDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import metrics
from torch.utils.data import DataLoader
from models.transformer import DIT
from diffusers import DDIMScheduler
import time

#************************************************************************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ldm = 'dit_vqvae_ckpt_name'
coder = 'vqvae_autoencoder_ckpt_name'


def DDIMsched(xt,model, conds, vae,ddim_scheduler):
    num_inference_steps = 50
    eta = 0
    ddim_scheduler.set_timesteps(num_inference_steps)

    #start_time = time.time()

    for i in ddim_scheduler.timesteps:
        timestep = i  # scalar int
        t_batch = torch.full((xt.shape[0],), timestep, device=xt.device, dtype=torch.long)
        noise_pred = model(xt, t_batch, conds)

        # Step to previous x_t-1
        step_output = ddim_scheduler.step(noise_pred, timestep, xt,eta=eta)
        xt = step_output.prev_sample

        if timestep == 0:
            ims = vae.decode(xt)
        else:
            ims = xt
    #total_time = time.time() - start_time  # <-- End timing
    #print(f"Sampling took {total_time:.3f} seconds for {num_inference_steps} steps")
    return ims

# ************************** Inference on test or real_world data *************************
def sample(model, vae,sample_scheduler, train_config, autoencoder_model_config, dataset_config,real_sampling=False):
    if real_sampling:     # generating CT for real-worl voltages
        hdf5_path = "path_to_real-world_voltages"
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            voltages = np.array(hdf5_file['voltages']).astype(np.float32)
        all_generated = []
        diffusion_dir = os.path.join(train_config['task_name'], 'real_data')
        os.makedirs(diffusion_dir, exist_ok=True)
        for idx, volt in enumerate(tqdm(voltages)):
            if idx >= 50:
                break
            volt = torch.tensor(volt).reshape(1, 208).to(device)
            im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
            xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size)).to(device)
            decoded = DDIMsched(xt, model, volt, vae,sample_scheduler)
            decoded = torch.clamp(decoded, 0., 1.).detach().cpu()
            all_generated.append(decoded)
            # Save generated image
            gen_img = decoded.squeeze().numpy()
            gen_path = os.path.join(diffusion_dir, f"sample_{idx}.png")
            plt.imsave(gen_path, gen_img, cmap="gray")
    else:       # generating CT for simulated test voltages
        im_dataset_cls = {
            'ct_cond': CtDataset
        }.get(dataset_config['name'])

        test_dataset = im_dataset_cls(split='test',
                                     test_path=dataset_config['test_path'],
                                     im_size=dataset_config['im_size'],
                                     im_channels=dataset_config['im_channels'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        diffusion_dir = os.path.join(train_config['task_name'], 'genCT')
        actual_dir = os.path.join(train_config['task_name'], 'actualCT')
        os.makedirs(diffusion_dir, exist_ok=True)
        os.makedirs(actual_dir, exist_ok=True)

        all_generated = []
        all_actual = []
        for idx, data in enumerate(tqdm(test_loader)):
            volts, images = data
            im = images.to(device)

            volts = volts.reshape(1, 208).to(device)
            im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
            seed = 0
            gen = torch.Generator(device).manual_seed(seed)
            xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size),generator=gen,device=device)
            decoded = DDIMsched(xt, model, volts, vae,sample_scheduler)
            decoded = torch.clamp(decoded, 0., 1.).detach().cpu()
            all_generated.append(decoded)
            all_actual.append(im)

            # Save generated image
            gen_img = decoded.squeeze().numpy()
            gen_path = os.path.join(diffusion_dir, f"sample_{idx}.png")
            plt.imsave(gen_path, gen_img, cmap="gray")

            # Save actual image
            real_img = im.squeeze().cpu().numpy()
            real_path = os.path.join(actual_dir, f"actual_{idx}.png")
            plt.imsave(real_path, real_img, cmap="gray")

# ************************** Evaluation during Training *************************

def Evaluate(model,vae,sample_scheduler,val_loader,train_config,autoencoder_model_config,dataset_config ):

    diffusion_dir = os.path.join(train_config['task_name'], 'valid_diffusion')
    actual_dir = os.path.join(train_config['task_name'], 'valid_actual')
    os.makedirs(diffusion_dir, exist_ok=True)
    os.makedirs(actual_dir, exist_ok=True)

    all_generated = []
    all_actual = []
    for idx, data in enumerate(tqdm(val_loader)):
        volts, images= data
        im = images.to(device)

        volts = volts.reshape(1,208).to(device)

        im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
        xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size)).to(device)
        decoded = DDIMsched(xt,model, volts,vae,sample_scheduler)
        decoded = torch.clamp(decoded, 0., 1.).detach().cpu()
        all_generated.append(decoded)
        all_actual.append(im)

        # Save generated image
        gen_img = decoded.squeeze().numpy()
        gen_path = os.path.join(diffusion_dir, f"sample_{idx}.png")
        plt.imsave(gen_path, gen_img, cmap="gray")

        # Save actual image
        real_img = im.squeeze().cpu().numpy()
        real_path = os.path.join(actual_dir, f"actual_{idx}.png")
        plt.imsave(real_path, real_img, cmap="gray")

    error, ssim = metrics.compute_metrics(torch.stack(all_actual).cpu().numpy().reshape(-1, 256, 256),
                            torch.stack(all_generated).cpu().numpy().reshape(-1, 256, 256))

    return error, ssim


# ********************************************************************************************


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='../config/ct.yaml', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    model = DIT(im_size=autoencoder_model_config['latent_size'],
                im_channels=autoencoder_model_config['z_channels'],
                config=dit_model_config).to(device)
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['dit_ldm_ckpt_name'])):
        print('Loading DIT checkpoints')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['dit_ldm_ckpt_name']), map_location=device))
    else:
        print('No saved model found. Initializing a new DIT model.')

    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)

    # Load vqvae if pretrained found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loading VQVAE checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device))
    else:
        print('No saved model found. Initializing a new VQVAE model.')
    model.eval()
    vae.eval()

    sample_scheduler = DDIMScheduler(
        num_train_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        clip_sample=False,
        set_alpha_to_one=False

    )

    sample(model, vae,sample_scheduler, train_config, autoencoder_model_config, dataset_config,real_sampling=False)

