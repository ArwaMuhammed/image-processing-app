import numpy as np
import cv2
from core.frequency import _gaussian_mask, _fft, _ifft


def _resize_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """Resize img2 to match img1's spatial dimensions."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2_resized


def _apply_gaussian_filter_freq(image: np.ndarray, cutoff: int, low_pass: bool) -> np.ndarray:
    """
    Apply a Gaussian low-pass or high-pass filter to a single channel (float32 [0,1]).
    Returns uint8 result.
    """
    is_color = len(image.shape) == 3

    if is_color:
        channels = cv2.split(image)
    else:
        channels = [image]

    shape = channels[0].shape
    mask = _gaussian_mask(shape, cutoff, low_pass)

    filtered = []
    for ch in channels:
        ch_float = ch.astype(np.float32) / 255.0
        f_shift = _fft(ch_float)
        f_filtered = f_shift * mask
        result = _ifft(f_filtered)
        filtered.append(result)

    if is_color:
        return cv2.merge(filtered)
    return filtered[0]


def create_hybrid_image(
    image1: np.ndarray,
    image2: np.ndarray,
    low_cutoff: int = 30,
    high_cutoff: int = 20,
    alpha: float = 0.5,
) -> tuple:
    """
    Create a hybrid image combining the low-frequency content of image1
    with the high-frequency content of image2.

    In hybrid images, up-close you see image2's high-frequency details,
    while from a distance you see image1's low-frequency structure.

    Parameters
    ----------
    image1      : BGR or grayscale image — contributes LOW frequencies.
    image2      : BGR or grayscale image — contributes HIGH frequencies.
    low_cutoff  : Gaussian LPF cutoff radius for image1.
    high_cutoff : Gaussian HPF cutoff radius for image2.
    alpha       : Blending weight in [0, 1].
                  Output = alpha * low + (1-alpha) * high.

    Returns
    -------
    hybrid      : uint8 hybrid image.
    low_pass_img: uint8 low-frequency component of image1.
    high_pass_img: uint8 high-frequency component of image2.
    """
    if image1 is None or image2 is None:
        raise ValueError("Both images must be provided.")

    # Match sizes and colour spaces
    image1, image2 = _resize_to_match(image1, image2)

    is_color1 = len(image1.shape) == 3
    is_color2 = len(image2.shape) == 3

    # Convert both to same type (colour preferred)
    if is_color1 and not is_color2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    elif is_color2 and not is_color1:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

    # ---- Low-pass image1 ----
    low_pass_img = _apply_gaussian_filter_freq(image1, cutoff=low_cutoff, low_pass=True)

    # ---- High-pass image2 ----
    high_pass_img = _apply_gaussian_filter_freq(image2, cutoff=high_cutoff, low_pass=False)

    # ---- Blend ----
    low_f  = low_pass_img.astype(np.float32)
    high_f = high_pass_img.astype(np.float32)

    hybrid_f = alpha * low_f + (1.0 - alpha) * high_f
    hybrid = np.clip(hybrid_f, 0, 255).astype(np.uint8)

    return hybrid, low_pass_img, high_pass_img


def visualize_hybrid_scales(hybrid: np.ndarray, scales: int = 5) -> list:
    """
    Return a list of progressively downscaled versions of the hybrid image
    to simulate the distance effect (useful for display in the UI).

    Parameters
    ----------
    hybrid : The hybrid image (uint8).
    scales : Number of scale levels.

    Returns
    -------
    List of images, index 0 is the original size.
    """
    results = [hybrid]
    current = hybrid.copy()
    for _ in range(scales - 1):
        h, w = current.shape[:2]
        if h < 20 or w < 20:
            break
        current = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        results.append(current)
    return results