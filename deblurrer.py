import numpy as np
import cv2
from skimage.transform import radon
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def estimate_blur_angle_frequency_debug(image, debug=False):
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    # Apply Hanning window to reduce edge effects
    hanning = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    windowed = gray * hanning

    # Compute FFT and magnitude spectrum
    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Smooth spectrum to suppress noise
    blurred_spectrum = gaussian_filter(magnitude_spectrum, sigma=3)

    # Compute gradients in frequency domain
    grad_x = cv2.Sobel(blurred_spectrum, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_spectrum, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    gradient_angle = (gradient_angle + 180) % 180  # Normalize

    # Histogram of orientations weighted by gradient magnitude
    hist, bin_edges = np.histogram(gradient_angle, bins=180, range=(0, 180), weights=gradient_magnitude)
    dominant_freq_angle = bin_edges[np.argmax(hist)]

    # Provide both the direct orientation and the offset 90° angle
    candidate_blur_angle_1 = dominant_freq_angle
    candidate_blur_angle_2 = (dominant_freq_angle + 90) % 180

    if debug:
        print(f"Dominant frequency orientation: {dominant_freq_angle:.2f}°")
        print(f"Candidate blur angles: {candidate_blur_angle_1:.2f}° and {candidate_blur_angle_2:.2f}°")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("FFT Magnitude Spectrum")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[:-1], hist)
        plt.title("Histogram of Frequency Orientations")
        plt.xlabel("Angle (°)")
        plt.ylabel("Weighted Count")
        plt.tight_layout()
        plt.savefig("debug1.png", dpi=300, bbox_inches='tight')

    return candidate_blur_angle_1, candidate_blur_angle_2

def generate_motion_blur_kernel(length, angle):
    size = length
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle_rad = np.deg2rad(angle)

    x1 = int(center + (size / 2) * np.cos(angle_rad))
    y1 = int(center + (size / 2) * np.sin(angle_rad))
    x2 = int(center - (size / 2) * np.cos(angle_rad))
    y2 = int(center - (size / 2) * np.sin(angle_rad))

    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    kernel /= np.sum(kernel)
    return kernel

def estimate_blur_kernel_with_dual_angle(image, kernel_length=60, debug=False):
    angle1, angle2 = estimate_blur_angle_frequency_debug(image, debug=debug)

    kernel1 = generate_motion_blur_kernel(length=kernel_length, angle=angle1)
    kernel2 = generate_motion_blur_kernel(length=kernel_length, angle=angle2)

    if debug:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(kernel1, cmap='gray')
        plt.title(f"Kernel (angle={angle1:.2f}°)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(kernel2, cmap='gray')
        plt.title(f"Kernel (angle={angle2:.2f}°)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("debug2.png", dpi=300, bbox_inches='tight')

    return (kernel1, angle1), (kernel2, angle2)


def main():
    # Load your blurred image (as color or grayscale)
    image_path = 'outputs/1_blurred.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
    (kernel1, angle1), (kernel2, angle2) = estimate_blur_kernel_with_dual_angle(image, kernel_length=100, debug=True)
    print(kernel1, angle1, kernel2, angle2)


if __name__ == "__main__":
    main()
