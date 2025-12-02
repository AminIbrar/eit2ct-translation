import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to calculate MSE
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Function to calculate PSNR
def calculate_psnr(original_image, reconstructed_image):
    mse = calculate_mse(original_image, reconstructed_image)
    max_pixel_value = 1.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse)) if mse > 0 else float('inf')

# Function to calculate Relative Image Error (RIE)
def relative_image_error(true_image, reconstructed_image):
    return np.linalg.norm(true_image - reconstructed_image) / np.linalg.norm(true_image)

# Function to calculate Correlation Coefficient (CC)
def correlation_coefficient(true_image, reconstructed_image):
    return np.corrcoef(true_image.flatten(), reconstructed_image.flatten())[0, 1]

# Function to compute and print all metrics
def compute_metrics(actual_images, pred_images):
    assert len(actual_images) == len(pred_images), "Mismatch in number of images!"

    mse_list, psnr_list, ssim_list, rie_list, cc_list = [], [], [], [], []

    for i in range(len(actual_images)):
        mse_list.append(calculate_mse(actual_images[i], pred_images[i]))
        psnr_list.append(calculate_psnr(actual_images[i], pred_images[i]))
        ssim_list.append(ssim(actual_images[i], pred_images[i], data_range=1.0))
        rie_list.append(relative_image_error(actual_images[i], pred_images[i]))
        cc_list.append(correlation_coefficient(actual_images[i], pred_images[i]))

    # Compute averages
    average_mse = np.mean(mse_list)
    average_psnr = np.mean(psnr_list)
    average_ssim = np.mean(ssim_list)
    average_rie = np.mean(rie_list)
    average_cc = np.mean(cc_list)

    return average_mse, average_ssim
