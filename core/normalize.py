import numpy as np

def normalize_image(image: np.ndarray) -> np.ndarray:

    img = image.astype(np.float64)
    
    # Find min and max values
    min_val = img.min()
    max_val = img.max()
    
    # Avoid division by zero if image is flat
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8) # returns a black image
    
    # Scale to 0-255 range: (pixel - min) / (max - min) * 255
    normalized = (img - min_val) / (max_val - min_val) * 255
    
    return normalized.astype(np.uint8)
