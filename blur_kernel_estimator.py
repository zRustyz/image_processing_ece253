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
    estimated_blur_angle = dominant_gradient_direction % 180

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

def estimate_blur_length_fourier(image, blur_angle_deg, debug=False):
    """
    Estimate the length (in pixels) of a linear motion blur,
    given its direction in the spatial domain.

    Parameters
    ----------
    image : np.ndarray
        Input image. Can be RGB or grayscale, float in [0,1] or uint8.
    blur_angle_deg : float
        Blur direction in the spatial domain, in degrees (0° = +x axis, CCW).
    debug : bool
        If True, plots the 1D Fourier profile and prints intermediate info.

    Returns
    -------
    length_est : float
        Estimated blur length in pixels.
    """

    # 1. Convert to grayscale and apply Hanning window
    if image.ndim == 3:
        # assume RGB
        if image.dtype != np.uint8:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        if image.dtype != np.uint8:
            gray = (image * 255).astype(np.uint8)
        else:
            gray = image

    gray = gray.astype(np.float32)
    h, w = gray.shape

    # 2D Hanning window to reduce edge artifacts in FFT
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    windowed = gray * window

    # 2. Fourier transform and log-magnitude spectrum
    F = np.fft.fft2(windowed)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift))  # log(1 + |F|)

    # Smooth spectrum to make oscillations cleaner
    mag_smooth = gaussian_filter(mag, sigma=3)

    # 3. Sample 1D profile along the blur direction in the frequency domain
    # DC is at the center after fftshift
    cy, cx = h // 2, w // 2

    # blur_angle_deg is the spatial blur direction; its frequency-domain
    # oscillations occur along the same axis.
    theta = np.deg2rad(blur_angle_deg)
    dir_vec = np.array([np.sin(theta), np.cos(theta)])  # (dy, dx)

    # max radius: half-diagonal of the image
    max_radius = int(0.5 * np.hypot(h, w))
    profile = []

    for r in range(1, max_radius):
        y = int(round(cy + r * dir_vec[0]))
        x = int(round(cx + r * dir_vec[1]))
        if y < 0 or y >= h or x < 0 or x >= w:
            break
        profile.append(mag_smooth[y, x])

    profile = np.array(profile, dtype=np.float32)
    if profile.size < 10:
        # Not enough samples to say anything meaningful
        if debug:
            print("Profile too short, returning length 1.")
        return 1.0

    # 4. Normalize and lightly smooth the 1D profile
    profile -= profile.min()
    denom = profile.max() + 1e-8
    profile /= denom

    # Simple 1D smoothing kernel
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    profile_smooth = np.convolve(profile, kernel, mode="same")

    # 5. Find the first significant local minimum (first "zero" of sinc-like pattern)
    # Skip a few samples near the center (DC region)
    start_idx = 3
    center_val = profile_smooth[start_idx]
    first_min_idx = None

    for i in range(start_idx + 1, len(profile_smooth) - 1):
        if profile_smooth[i] < profile_smooth[i - 1] and profile_smooth[i] < profile_smooth[i + 1]:
            # Require the valley to be noticeably below the DC region
            if profile_smooth[i] < 0.9 * center_val:
                first_min_idx = i
                break

    if first_min_idx is None or first_min_idx <= 0:
        # Fallback if no clear minimum found
        length_est = 1.0
    else:
        # 6. Convert frequency-zero spacing → blur length
        # For a rect blur of length L, zeros appear at f ≈ n / L.
        # With DFT sampling step Δf ≈ 1 / N_eff, first zero at bin k1 gives:
        #   f1 ≈ k1 / N_eff  ≈ 1 / L  →  L ≈ N_eff / k1
        N_eff = min(h, w)
        length_est = float(N_eff) / float(first_min_idx)

    if debug:
        print(f"First minimum index: {first_min_idx}")
        print(f"Estimated blur length: {length_est:.2f} pixels")

        x_axis = np.arange(len(profile_smooth))
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, profile_smooth, label="1D Fourier profile")
        if first_min_idx is not None:
            plt.axvline(first_min_idx, linestyle="--", label="First significant minimum")
        plt.xlabel("Radius in frequency bins")
        plt.ylabel("Normalized magnitude")
        plt.title(f"Profile along blur direction (angle = {blur_angle_deg:.1f}°)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("debug_blur_length_profile.png", dpi=300, bbox_inches="tight")

    return int(length_est)

def main():
    # Load your blurred image (as color or grayscale)
    image_path = 'outputs/5/blurry.png'
    # image_path = 'outputs/5_noisy.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
    angle = estimate_blur_angle_directional(image, debug=True)
    # length = estimate_blur_length_fft_peaks(image, 180, debug=True)
    # length = estimate_blur_length_spatial(image, 180)
    length = estimate_blur_length_fourier(image, angle)
    print(angle)
    print(length)


if __name__ == "__main__":
    main()

