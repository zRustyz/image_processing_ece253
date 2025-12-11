import numpy as np
import cv2
from blur_kernel_estimator import estimate_blur_angle_directional
from noise_to_signal_estimator import nsr_blind_estimator
from degrader import motion_blur_kernel


def wiener_deblur_color(image, kernel, nsr, output_path="outputs/wiener_result.png"):
    """
    Apply Wiener deblurring to a color image and save the result.

    Parameters:
    - image: Blurred + noisy color image (H, W, 3), float32, range [0, 1]
    - kernel: Blur kernel (2D numpy array)
    - nsr: Noise-to-signal ratio (float)
    - output_path: File path to save the output PNG

    Returns:
    - Deblurred image (H, W, 3), float32, clipped to [0, 1]
    """
    image = np.clip(image, 0, 1)
    kernel /= np.sum(kernel)
    pad_kernel = np.zeros_like(image[:, :, 0])
    kh, kw = kernel.shape
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -kh//2, axis=0)
    pad_kernel = np.roll(pad_kernel, -kw//2, axis=1)
    H = np.fft.fft2(pad_kernel)

    result = np.zeros_like(image)
    for c in range(3):  # RGB channels
        I = np.fft.fft2(image[:, :, c])
        H_conj = np.conj(H)
        denominator = (np.abs(H) ** 2 + nsr)
        F_hat = (H_conj / denominator) * I
        result[:, :, c] = np.abs(np.fft.ifft2(F_hat))

    result = np.clip(result, 0, 1)
    cv2.imwrite(output_path, (result * 255).astype(np.uint8)[..., ::-1])
    return result

def inverse_deblur_color(image, kernel, epsilon=1e-3, output_path="outputs/inverse_result.png"):
    """
    Apply inverse deblurring to a color image using consistent kernel alignment.

    Parameters:
    - image: Blurred color image (H, W, 3), float32, range [0, 1]
    - kernel: Blur kernel (2D numpy array)
    - epsilon: Small constant to prevent division by zero
    - output_path: File path to save the output PNG

    Returns:
    - Deblurred image (H, W, 3), float32, clipped to [0, 1]
    """
    image = np.clip(image, 0, 1)
    kernel /= np.sum(kernel)

    h, w = image.shape[:2]
    pad_kernel = np.zeros((h, w), dtype=np.float32)
    kh, kw = kernel.shape
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -kh // 2, axis=0)
    pad_kernel = np.roll(pad_kernel, -kw // 2, axis=1)
    H = np.fft.fft2(pad_kernel)

    restored = np.zeros_like(image)
    for c in range(3):
        B = np.fft.fft2(image[:, :, c])
        F_hat = B / (H + epsilon)
        restored[:, :, c] = np.real(np.fft.ifft2(F_hat))

    restored = np.clip(restored, 0, 1)
    cv2.imwrite(output_path, (restored * 255).astype(np.uint8)[..., ::-1])
    return restored


def main():
    
    for i in range(1, 8):
        image_path = f"outputs/{i}/noisy.png"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
        angle = estimate_blur_angle_directional(image, debug=True)
        nsr_estimate = nsr_blind_estimator(image, patch_size=8, top_k_percent=10, gradient_threshold=5.0)
        kernel = motion_blur_kernel(length=27, angle=angle)
        wiener_deblur_color(image, kernel, nsr_estimate, output_path=f"outputs/{i}/wiener_result_noisy.png")
        inverse_deblur_color(image, kernel, output_path=f"outputs/{i}/inverse_result_noisy.png")
    print("saved image to outputs/")

if __name__ == "__main__":
    main()
