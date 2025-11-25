import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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

    # Rotate by 90° to convert frequency suppression axis to motion blur direction
    estimated_blur_angle = (dominant_gradient_direction + 90) % 360

    if debug:
        print(f"Dominant gradient direction: {dominant_gradient_direction:.2f}°")
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


def main():
    # Load your blurred image (as color or grayscale)
    image_path = 'outputs/1_blurred.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
    angle = estimate_blur_angle_directional(image, debug=True)
    print(angle)


if __name__ == "__main__":
    main()

