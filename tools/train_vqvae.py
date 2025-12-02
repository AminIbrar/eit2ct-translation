import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data import DataLoader
from dataset.dataset import CtDataset
from torch.optim import Adam
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    ############## Loss Flags ###################################
    use_adversarial = train_config.get('use_adversarial', True)
    use_lpips = train_config.get('use_lpips', True)
    use_mse = train_config.get('use_mse', True)

    print(f"Training with - Adversarial: {use_adversarial}, LPIPS: {use_lpips}, MSE: {use_mse}")
    ################ Model and Dataset ############################

    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    # Load vqvae if pretrained found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loading VQAE checkpoints')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['vqvae_autoencoder_ckpt_name']),
                                         map_location=device))
    else:
        print('No saved model found. Initializing a new VQAE model.')
    # Create the dataset
    im_dataset_cls = {
        'ct_cond': CtDataset
    }.get(dataset_config['name'])

    # Create train and validation datasets
    train_dataset = im_dataset_cls(split='train',
                                   train_path=dataset_config['train_path'],
                                   val_path=None,
                                   im_size=dataset_config['im_size'],
                                   im_channels=dataset_config['im_channels'])

    val_dataset = im_dataset_cls(split='val',
                                 train_path=None,
                                 val_path=dataset_config['val_path'],
                                 im_size=dataset_config['im_size'],
                                 im_channels=dataset_config['im_channels'])

    # Create dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=train_config['autoencoder_batch_size'],
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=train_config['autoencoder_batch_size'],
                            shuffle=False)  # No need to shuffle validation set
    ##############################################################################
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['autoencoder_epochs']

    # CONDITIONALLY initialize Losses
    recon_criterion = torch.nn.MSELoss() if use_mse else None
    disc_criterion = torch.nn.MSELoss() if use_adversarial else None

    # Initialize LPIPS and Discriminator
    lpips_model = None
    if use_lpips:
        lpips_model = LPIPS().eval().to(device)
    discriminator = None
    optimizer_d = None
    if use_adversarial:
        discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
        optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    disc_step_start = train_config['disc_start'] if use_adversarial else float('inf')
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    best_val_loss = float("inf")
    for epoch_idx in range(train_config['autoencoder_epochs']):

        recon_losses = []
        codebook_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad()
        if optimizer_d:
            optimizer_d.zero_grad()

        for volt, im in tqdm(train_loader):
            step_count += 1
            im = im.to(device)

            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output

            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], 0., 1.).detach().cpu()
                save_input = im[:sample_size].detach().cpu()

                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()

            ######### Optimize Generator ##########
            g_loss = 0

            # MSE Reconstruction Loss
            if use_mse:
                recon_loss = recon_criterion(output, im)
                recon_losses.append(recon_loss.item())
                recon_loss_scaled = recon_loss / acc_steps
                g_loss += recon_loss_scaled

            # Codebook and Commitment Losses (always used in VQ-VAE)
            codebook_loss = (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps)
            commitment_loss = (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps)
            g_loss += codebook_loss + commitment_loss
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # Adversarial Loss
            if use_adversarial and step_count > disc_step_start and discriminator:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_loss_val = train_config['disc_weight'] * disc_fake_loss / acc_steps
                g_loss += gen_loss_val
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())

            # Perceptual (LPIPS) Loss
            if use_lpips and lpips_model:
                lpips_loss = torch.mean(lpips_model(output, im))
                lpips_loss_scaled = train_config['perceptual_weight'] * lpips_loss / acc_steps
                g_loss += lpips_loss_scaled
                perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())

            losses.append(g_loss.item())
            g_loss.backward()
            #####################################

            ######### Optimize Discriminator #######
            if use_adversarial and step_count > disc_step_start and discriminator:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0 and optimizer_d:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
                if optimizer_d:
                    optimizer_d.zero_grad()

        # Final optimizer steps at end of epoch
        if step_count % acc_steps != 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
            if optimizer_d and use_adversarial and step_count > disc_step_start:
                optimizer_d.step()
                optimizer_d.zero_grad()

        # Dynamic printing based on active losses
        log_parts = [f'Epoch: {epoch_idx + 1}']
        if recon_losses:
            log_parts.append(f'Recon: {np.mean(recon_losses):.4f}')
        if perceptual_losses:
            log_parts.append(f'LPIPS: {np.mean(perceptual_losses):.4f}')
        if codebook_losses:
            log_parts.append(f'Codebook: {np.mean(codebook_losses):.4f}')
        if gen_losses:
            log_parts.append(f'G Loss: {np.mean(gen_losses):.4f}')
        if disc_losses:
            log_parts.append(f'D Loss: {np.mean(disc_losses):.4f}')

        print(' | '.join(log_parts))

        # Validation
        if epoch_idx % 1 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _, im in tqdm(val_loader):
                    im = im.float().to(device)
                    model_output = model(im)
                    output, _, _ = model_output
                    # Use MSE for validation regardless of training config
                    recon_loss = torch.nn.MSELoss()(output, im)
                    val_loss += recon_loss.item()
                print(f'--------- Validation loss is {val_loss / len(val_loader)} -----------')

        # Model saving
        print('Saving Best validation model ....')
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        # Only save discriminator if used
        if use_adversarial and discriminator:
            torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['vqvae_discriminator_ckpt_name']))
        model.train()

    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='../config/ct.yaml', type=str)
    args = parser.parse_args()
    train(args)