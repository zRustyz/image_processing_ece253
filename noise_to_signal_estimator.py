import numpy as np
import cv2

def nsr_test(noisy_image, clean_image, patch_size=8):
    """
    Compare estimated NSR from noisy image only vs. true NSR using clean image.

    Parameters:
    - noisy_image: RGB or grayscale noisy image (normalized to [0, 1])
    - clean_image: Clean reference image (same shape and dtype as noisy_image)
    - patch_size: Size of square patches for noise estimation

    Returns:
    - nsr_true: Ground-truth noise-to-signal ratio
    - nsr_estimated: Estimated NSR from noisy image only
    """
    # === Ensure proper format and grayscale conversion
    noisy_image = np.asarray(noisy_image)
    clean_image = np.asarray(clean_image)

    if noisy_image.ndim == 3 and noisy_image.shape[2] == 3:
        noisy_gray = cv2.cvtColor((noisy_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        clean_gray = cv2.cvtColor((clean_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        noisy_gray = (noisy_image * 255).astype(np.uint8)
        clean_gray = (clean_image * 255).astype(np.uint8)

    # === Compute true NSR
    noise = noisy_gray.astype(np.float32) - clean_gray.astype(np.float32)
    noise_power = np.mean(noise ** 2)
    signal_power = np.mean(clean_gray.astype(np.float32) ** 2)
    nsr_true = noise_power / signal_power if signal_power != 0 else 0.0

    # === Estimate NSR from noisy image only
    h, w = int(noisy_gray.shape[0]), int(noisy_gray.shape[1])
    variances = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = noisy_gray[y:y+patch_size, x:x+patch_size]
            patch_var = np.var(patch.astype(np.float32))
            variances.append(patch_var)

    if len(variances) == 0:
        nsr_estimated = 0.0
    else:
        variances = np.array(variances)
        noise_var_est = np.median(np.sort(variances)[:len(variances) // 10])
        image_var_est = np.var(noisy_gray.astype(np.float32))
        nsr_estimated = noise_var_est / image_var_est if image_var_est != 0 else 0.0

    return nsr_true, nsr_estimated

def nsr_blind_estimator(noisy_image, patch_size=8, top_k_percent=10, gradient_threshold=2.0):
    """
    Improved blind NSR estimation from a noisy image using gradient and trimmed variance filtering.

    Parameters:
    - noisy_image: RGB or grayscale image normalized to [0, 1]
    - patch_size: Size of square patches
    - top_k_percent: Percent of flattest patches used (default 10%)
    - gradient_threshold: Max average gradient magnitude to consider a patch flat

    Returns:
    - Estimated NSR (float)
    """
    noisy_image = np.asarray(noisy_image)

    # Convert to grayscale if needed
    if noisy_image.ndim == 3 and noisy_image.shape[2] == 3:
        gray = cv2.cvtColor((noisy_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (noisy_image * 255).astype(np.uint8)

    h, w = int(gray.shape[0]), int(gray.shape[1])
    patch_vars = []
    gradients = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size].astype(np.float32)
            grad_patch = gradients[y:y+patch_size, x:x+patch_size]
            mean_grad = np.mean(np.abs(grad_patch))

            if mean_grad < gradient_threshold:  # consider only truly flat patches
                patch_var = np.var(patch)
                patch_vars.append(patch_var)

    if not patch_vars:
        return 0.0

    patch_vars = np.array(sorted(patch_vars))
    k = max(1, int(len(patch_vars) * top_k_percent / 100))
    noise_var = np.mean(patch_vars[:k])
    total_var = np.var(gray.astype(np.float32))

    return noise_var / total_var if total_var != 0 else 0.0


if __name__ == "__main__":

    # Load your noisy and (optionally) clean image
    noisy = cv2.imread("outputs/1_noisy.png")
    noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB) / 255.0

    # Estimate NSR from noisy image only
    nsr_estimated = nsr_blind_estimator(noisy, patch_size=8, top_k_percent=10, gradient_threshold=5.0)
    print(f"[Estimated from noisy image only] NSR: {nsr_estimated:.6f}")

    # If you also have the clean image
    clean = cv2.imread("outputs/1_blurred.png")
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB) / 255.0
    nsr_true, nsr_estimated = nsr_test(noisy, clean)
    print(f"Ground Truth NSR     : {nsr_true:.6f}")
    print(f"Estimated NSR (blind): {nsr_estimated:.6f}")

