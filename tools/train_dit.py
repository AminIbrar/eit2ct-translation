import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from dataset.dataset import CtDataset
from torch.utils.data import DataLoader
from models.transformer import DIT
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from diffusers import DDIMScheduler
from sample_dit import Evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dit_model_config = config['dit_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    train_scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    sample_scheduler = DDIMScheduler(
        num_train_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        clip_sample=False,
        set_alpha_to_one=False

    )

    train_dataset = CtDataset(split='train',
                              train_path=dataset_config['train_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels']

                              )
    val_dataset = CtDataset(split='val',
                            val_path=dataset_config['val_path'],
                            im_size=dataset_config['im_size'],
                            im_channels=dataset_config['im_channels']

                            )
    train_loader = DataLoader(train_dataset,
                              batch_size=train_config['ldm_batch_size'],
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False)

    # *****************************************************************************************
    model = DIT(im_size=autoencoder_model_config['latent_size'],
                im_channels=autoencoder_model_config['z_channels'],
                config=dit_model_config).to(device)

    # Load checkpoint for DIT model
    if os.path.exists(os.path.join(train_config['task_name'], train_config['dit_ldm_ckpt_name'])):
        print('Loading DIT checkpoints')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['dit_ldm_ckpt_name']), map_location=device))
    else:
        print("No saved model found. Initialized a new ddpm model.")
    # ************************************************************************************
    # create instance of autoencoder
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)

    # Load vqvae if pretrained found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loading VQAE checkpoints')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device))
    else:
        print('No saved model found. Initializing a new VQAE model.')
    vae.eval()



    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=3e-3)
    criterion = torch.nn.MSELoss()
    # Run training
    for epoch_idx in range(num_epochs):
        model.train()
        noise_loss = []
        for batch_idx, data in enumerate(tqdm(train_loader)):
            volts, images= data
            images = images.to(device)
            volts = volts.to(device)
            optimizer.zero_grad()
            vae.eval()
            with torch.no_grad():
                im, _ = vae.encode(images)
            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            noisy_im = train_scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, volts)
            n_loss = criterion(noise_pred, noise)
            noise_loss.append(n_loss.item())
            n_loss.backward()
            optimizer.step()

        avg_train_noise_loss = np.mean(noise_loss)
        print('Finished epoch:{} | Train_Noise_Loss : {:.4f} '.format(epoch_idx + 1, avg_train_noise_loss))

        # *********************** Validation ********************************
        if epoch_idx % 5 == 0:
            model.eval()
            vae.eval()
            error, ssm = Evaluate(model,vae,sample_scheduler,val_loader,train_config,autoencoder_model_config,dataset_config )
            val_noise_loss = []
            with torch.no_grad():
                for data in val_loader:
                    volts, images = data
                    images = images.float().to(device)
                    volts = volts.float().to(device)
                    im, _ = vae.encode(images)
                    noise = torch.randn_like(im).to(device)
                    t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
                    noisy_im = train_scheduler.add_noise(im, noise, t)
                    noise_pred = model(noisy_im, t, volts)
                    v_n_loss = criterion(noise_pred, noise)
                    val_noise_loss.append(v_n_loss.item())

            avg_val_noise_loss = np.mean(val_noise_loss)
            print(f'🧪 Validation Noise Loss: {avg_val_noise_loss:.4f} ')
            print(f'🧪 Sampling Error: {error:.4f} | SSIM: {ssm:.4f} ')

            print("✅ Saving model...")
            torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['dit_ldm_ckpt_name']))

    print('Done Training ...')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='../config/ct.yaml', type=str)
    args = parser.parse_args()
    train(args)
