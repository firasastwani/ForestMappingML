"""
Image preprocessing utilities for forest maps
Handles loading, cropping, preprocessing, and normilization of arial imagery
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class ImageProcessor:

    """Handles image preprocessing for forest classification images"""

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load the image from the file path"""

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found {image_path}")

        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to HSV for better computer imagery (justifiy in README)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img_hsv, img_rgb

    
    def crop_image(self, image: np.ndarray, crop_coords: Tuple[int, int, int, int]) -> np.ndarray:

        """Crop the image using the coords
            This is done to get rid of the unncessary data that might pollute the info
        """

        x1, y1, x2, y2 = crop_coords

        # y coords come before x coords
        return image[y1:y2, x1:x2]

    
    def resize_image(self, image: np.ndarray) -> np.ndarray:

        return cv2.resize(image, self.target_size)


    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if len(img.shape) == 2:
            # Already grayscale
            return img
        else:
            # Convert RGB to grayscale using luminance
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            return gray.astype(np.uint8)
    

    def enhance_image(self, img: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """Apply image enhancement techniques"""

        #Defualts to clahe, can add more in the future to test different enhancement methods. 
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


        # Swithcing on and off to test use of Gaussian Blur along with enhancement
        if False:
            cv2.GaussianBlur(img, (5, 5), 0)
            
        return clahe.apply(img)
    
    # important for ML
    def normalize_image(self, img: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize image for consistent processing"""
        if method == 'minmax':
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            return img
        elif method == 'zscore':
            mean, std = img.mean(), img.std()
            if std > 0:
                normalized = (img - mean) / std
                return ((normalized - normalized.min()) / 
                       (normalized.max() - normalized.min()) * 255).astype(np.uint8)
            return img
        elif method == 'percentile':
            # More robust normalization using percentiles
            p2, p98 = np.percentile(img, (2, 98))
            if p98 > p2:
                img_clipped = np.clip(img, p2, p98)
                return ((img_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
            return img
        else:
            return img
    
    def preprocess_image(self, image_path: Path, crop_coords: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    
        """Full preprocessing pipeline, returns tuple:
            HSV_IMAGE, RGB_IMAGE
            HSV for analysis
            RGB for display
        """

        # load the img in both hsv and rgb
        img_hsv, img_rgb = self.load_image(image_path)

        if crop_coords:
            img_hsv = self.crop_image(img_hsv, crop_coords)
            img_rgb= self.crop_image(img_rgb, crop_coords)


        img_hsv = self.resize_image(img_hsv)
        img_rgb = self.resize_image(img_rgb)



        return img_hsv, img_rgb
        

        
 
    