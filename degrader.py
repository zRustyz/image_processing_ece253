import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import convolve2d

# Load a color image and normalize 
def load_color_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return img

# Create a linear motion blur kernel
def motion_blur_kernel(length=100, angle=180):
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2

    # Define line endpoints based on the desired angle
    angle_rad = np.deg2rad(angle)
    x1 = int(center + (length // 2) * np.cos(angle_rad))
    y1 = int(center + (length // 2) * np.sin(angle_rad))
    x2 = int(center - (length // 2) * np.cos(angle_rad))
    y2 = int(center - (length // 2) * np.sin(angle_rad))

    # Draw line on kernel
    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    kernel /= np.sum(kernel)
    return kernel
'''
# Apply motion blur to each RGB channel
def apply_motion_blur_color(img, kernel):
    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[:, :, c] = convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
    return np.clip(blurred, 0, 1)
'''
def apply_motion_blur_color(image, kernel):
    """
    Apply motion blur to a color image using circular convolution in the frequency domain.
    """
    h, w = image.shape[:2]
    kernel = kernel / np.sum(kernel)

    # Pad and center the kernel inside the padded array
    padded_kernel = np.zeros((h, w), dtype=np.float32)
    kh, kw = kernel.shape
    y_offset = (h - kh) // 2
    x_offset = (w - kw) // 2
    padded_kernel[y_offset:y_offset+kh, x_offset:x_offset+kw] = kernel
    padded_kernel = np.fft.ifftshift(padded_kernel)

    H = np.fft.fft2(padded_kernel)

    blurred = np.zeros_like(image)
    for c in range(3):
        F = np.fft.fft2(image[:, :, c])
        B = np.fft.ifft2(F * H).real
        blurred[:, :, c] = np.clip(B, 0, 1)

    return blurred

# Add Gaussian noise to each RGB channel
def add_gaussian_noise_color(img, std_dev=0.01):
    noise = np.random.normal(0, std_dev, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

'''
# Save blurred, and noisy image
def show_color_images(original, blurred, noisy):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title('Original Image')
    axs[1].imshow(blurred)
    axs[1].set_title('Motion Blurred Image')
    axs[2].imshow(noisy)
    axs[2].set_title('Blurred + Noisy Image')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/1_degradedv3.png", dpi=300, bbox_inches='tight')
'''

def save_images(original, blurred, noisy, output):

    # Convert from float32 [0,1] to uint8 [0,255] and from RGB to BGR for OpenCV
    def to_bgr_uint8(img):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"outputs/{output}/original.png", (original * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(f"outputs/{output}/blurry.png", (blurred * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(f"outputs/{output}/noisy.png", (noisy * 255).astype(np.uint8)[..., ::-1])


# Main execution
if __name__ == "__main__":
    '''
    for i in range(1, 8):
        image_path = f"images/{i}.jpg"
        original = load_color_image(image_path)
        ang = random.randrange(180)
        print(ang)
        kernel = motion_blur_kernel(length=27, angle=ang)
        blurred = apply_motion_blur_color(original, kernel)
        noisy = add_gaussian_noise_color(blurred, std_dev=0.02)
        save_images(original, blurred, noisy, i)
    '''
    for i in range(1, 8):
        image_path = f"images/{i}.jpg"
        original = load_color_image(image_path)
        ang = random.randrange(180)
        print(ang)
        len = random.randrange(15)
        print(len)
        kernel = motion_blur_kernel(length=15, angle=0)
        blurred = apply_motion_blur_color(original, kernel)
        noisy = add_gaussian_noise_color(blurred, std_dev=0.02)
        save_images(original, blurred, noisy, i)
    print("saved image to outputs/")
