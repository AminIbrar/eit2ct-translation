import argparse
import os
import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from dataset.dataset import CtDataset
from models.vqvae import VQVAE
from utils import metrics
import time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the dataset
    im_dataset_cls = {
        'ct_cond': CtDataset
    }.get(dataset_config['name'])

    test_dataset = im_dataset_cls(split='test',
                                 test_path=dataset_config['val_path'],
                                 im_size=dataset_config['im_size'],
                                 im_channels=dataset_config['im_channels'])

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False)

    num_images = train_config['num_samples']

    # Create output directory
    output_dir_act = os.path.join(train_config['task_name'], 'actualCT')
    os.makedirs(output_dir_act, exist_ok=True)
    output_dir_gen = os.path.join(train_config['task_name'], 'vqvae_valid')
    os.makedirs(output_dir_gen, exist_ok=True)

    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['vqvae_autoencoder_ckpt_name']),
                                     map_location=device))
    model.eval()
    actual_ims = []
    decoded_ims = []
    with torch.no_grad():
        # Process images sequentially from the beginning
        for i, (_, batch) in enumerate(test_loader):
            ims = batch.float().to(device)

            encoded_output, _ = model.encode(ims)

            decoded_output = model.decode(encoded_output)
            encoded_output = torch.clamp(encoded_output, 0., 1.)
            decoded_output = torch.clamp(decoded_output, 0., 1.)

            # Save individual images
            input_img = torchvision.transforms.ToPILImage()(ims.cpu().squeeze(0))
            encoded_img = torchvision.transforms.ToPILImage()(encoded_output.cpu().squeeze(0))
            reconstructed_img = torchvision.transforms.ToPILImage()(decoded_output.cpu().squeeze(0))

            input_img.save(os.path.join(output_dir_act, f'input_{i:04d}.png'))
            reconstructed_img.save(os.path.join(output_dir_gen, f'reconstructed_{i:04d}.png'))

            actual_ims.append(ims.cpu().numpy().reshape(256, 256))
            decoded_ims.append(decoded_output.cpu().numpy().reshape(256, 256))


    metrics.compute_metrics(actual_ims, decoded_ims)
    print(f"Inference completed! Results saved to: {output_dir_gen}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='../config/ct.yaml', type=str)
    args = parser.parse_args()
    infer(args)