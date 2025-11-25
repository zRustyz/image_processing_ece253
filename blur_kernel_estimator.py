import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


def estimate_blur_angle_directional(image, debug=False):
    # Convert to grayscale if RGB
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    # Apply Hanning window to suppress edge artifacts
    hanning = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    windowed = gray * hanning

    # Compute FFT and get log-magnitude spectrum
    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Smooth the spectrum to reduce noise
    blurred_spectrum = gaussian_filter(magnitude_spectrum, sigma=3)

    # Compute gradients
    grad_x = cv2.Sobel(blurred_spectrum, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_spectrum, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    gradient_angle = (gradient_angle + 360) % 360  # Normalize to [0, 360)

    # Histogram of orientation weighted by gradient strength
    hist, bin_edges = np.histogram(gradient_angle, bins=360, range=(0, 360), weights=gradient_magnitude)
    dominant_gradient_direction = bin_edges[np.argmax(hist)]
    '''
    # Rotate by 90° to convert frequency suppression axis to motion blur direction
    estimated_blur_angle = (dominant_gradient_direction + 90) % 360
    '''
    estimated_blur_angle = dominant_gradient_direction

    if debug:
        # print(f"Dominant gradient direction: {dominant_gradient_direction:.2f}°")
        print(f"Estimated motion blur angle: {estimated_blur_angle:.2f}°")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("FFT Magnitude Spectrum")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[:-1], hist)
        plt.title("Gradient Orientation Histogram (0°–360°)")
        plt.xlabel("Angle (°)")
        plt.ylabel("Weighted Gradient Count")
        plt.tight_layout()
        plt.savefig("debug1.png", dpi=300, bbox_inches='tight')
    return estimated_blur_angle


def estimate_blur_length_fft_peaks(image, angle_deg, max_length=100, debug=False):
    """
    Estimate blur length using cepstrum analysis enhanced with peak filtering.

    Parameters:
    - image: Input image (RGB or grayscale), float32 in range [0, 1]
    - angle_deg: Known blur angle in degrees
    - max_length: Max length to search (int)
    - debug: If True, outputs debug visualizations

    Returns:
    - Estimated blur length (int)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    h, w = gray.shape

    # Window to reduce edge effects
    hanning = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray * hanning

    # FFT -> log spectrum -> cepstrum
    fft = np.fft.fft2(windowed)
    log_spectrum = np.log(np.abs(fft) + 1e-8)
    cepstrum = np.abs(np.fft.ifft2(log_spectrum))
    cepstrum = np.fft.fftshift(cepstrum)

    # Sample profile in blur direction
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    cx, cy = w // 2, h // 2

    profile = []
    for r in range(1, max_length):
        x = int(cx + r * dx)
        y = int(cy + r * dy)
        if 0 <= x < w and 0 <= y < h:
            profile.append(cepstrum[y, x])
        else:
            break

    profile = np.array(profile)
    smooth_profile = gaussian_filter1d(profile, sigma=2)

    estimated_length = np.argmax(smooth_profile[1:]) + 1  # skip DC

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cepstrum, cmap='gray')
        plt.title("Cepstrum")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(smooth_profile)), smooth_profile[1:])
        plt.axvline(estimated_length, color='r', linestyle='--', label=f'Length = {estimated_length}')
        plt.title("Blur Length Profile")
        plt.xlabel("Distance (px)")
        plt.ylabel("Cepstrum Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig("blur_length_peak_debug.png", dpi=300)

    return estimated_length

def estimate_blur_length_spatial(gray_image, angle_deg, max_length=100, visualize=False):
    """
    Estimate motion blur length along a given angle using directional autocorrelation.

    Parameters:
    - gray_image: 2D numpy array, grayscale image normalized to [0, 1]
    - angle_deg: float, known blur direction in degrees (e.g., 180 for horizontal)
    - max_length: int, maximum blur length to test
    - visualize: bool, if True shows correlation plot

    Returns:
    - Estimated blur length (int)
    """
    if gray_image.ndim == 3:
        gray = cv2.cvtColor((gray_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (gray_image * 255).astype(np.uint8)

    gray = gray.astype(np.float32)
    h, w = gray.shape

    corr = correlate2d(gray, gray, mode='full')
    center = np.array(corr.shape) // 2

    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    profile = []
    for l in range(1, max_length + 1):
        x = int(center[1] + l * dx)
        y = int(center[0] + l * dy)
        if 0 <= y < corr.shape[0] and 0 <= x < corr.shape[1]:
            profile.append(corr[y, x])
        else:
            break

    profile = np.array(profile)
    profile -= np.min(profile)
    profile /= np.max(profile) if np.max(profile) > 0 else 1

    est_length = np.argmax(profile[1:]) + 1

    if visualize:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(profile)+1), profile)
        plt.title("Autocorrelation Profile along Blur Direction")
        plt.xlabel("Blur Length")
        plt.ylabel("Normalized Correlation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return est_length

def main():
    # Load your blurred image (as color or grayscale)
    image_path = 'outputs/5_blurred.png'
    # image_path = 'outputs/5_noisy.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
    angle = estimate_blur_angle_directional(image, debug=True)
    # length = estimate_blur_length_fft_peaks(image, 180, debug=True)
    length = estimate_blur_length_spatial(image, 180)
    print(angle)
    print(length)


if __name__ == "__main__":
    main()

