import numpy as np
import cv2


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Pad image with zeros
    padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Use sliding window view for vectorized convolution (no Python loops)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (k_h, k_w))  # shape: (img_h, img_w, k_h, k_w)

    return np.einsum('ijkl,kl->ij', windows, np.flipud(np.fliplr(kernel)))


def sobel_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    # Sobel kernels
    # Kernel for detecting horizontal edges (gradient in x direction)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # Kernel for detecting vertical edges (gradient in y direction)
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    # Apply convolution with Sobel kernels
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    return gradient_magnitude


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
   
    return cv2.Canny(image, low_threshold, high_threshold)


def prewitt_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    # Prewitt kernels (3x3)
    # Kernel for horizontal edges (gradient in x direction)
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # Kernel for vertical edges (gradient in y direction)
    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    # Apply convolution with Prewitt kernels
    gradient_x = convolve(image, prewitt_x)
    gradient_y = convolve(image, prewitt_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    else:
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    return gradient_magnitude


def roberts_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    img_h, img_w = image.shape
    
    # Roberts Cross kernels (2x2)
    # These are applied differently - we handle the 2x2 case manually
    roberts_x = np.array([
        [1,  0],
        [0, -1]
    ], dtype=np.float64)
    
    roberts_y = np.array([
        [ 0, 1],
        [-1, 0]
    ], dtype=np.float64)
    
    # Initialize output arrays
    gradient_x = np.zeros((img_h, img_w), dtype=np.float64)
    gradient_y = np.zeros((img_h, img_w), dtype=np.float64)
    
    # Apply Roberts Cross operator (2x2 kernel, no padding needed for valid region)
    for i in range(img_h - 1):
        for j in range(img_w - 1):
            # Extract 2x2 region
            region = image[i:i+2, j:j+2]
            # Apply Roberts kernels
            gradient_x[i, j] = np.sum(region * roberts_x)
            gradient_y[i, j] = np.sum(region * roberts_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    else:
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    return gradient_magnitude
