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
            print('Already in greyscale')
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
    
    def normalize_image(self, img: np.ndarray, method: str = 'percentile') -> np.ndarray:
        """Normalize image for consistent processing"""
        if method == 'percentile':
            # More robust normalization using percentiles
            p2, p98 = np.percentile(img, (2, 98))
            if p98 > p2:
                img_clipped = np.clip(img, p2, p98)
                return ((img_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
            return img
        else:
            return img
    
    
    def preprocess_pipeline(self, image_path: Path, crop_coords: Optional[Tuple[int, int, int, int]] = None, 
                           enhance: bool = True, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
        """Full preprocessing pipeline, returns tuple:
            HSV_IMAGE, RGB_IMAGE
            HSV for analysis
            RGB for display
            Greyscale: For texture-based methods
        """

        # Step 1: Load image in both HSV and RGB
        img_hsv, img_rgb = self.load_image(image_path)
        
        # Step 2: Crop if coordinates provided
        if crop_coords:
            img_hsv = self.crop_image(img_hsv, crop_coords)
            img_rgb = self.crop_image(img_rgb, crop_coords)
        
        # Step 3: Resize to target size
        img_hsv = self.resize_image(img_hsv)
        img_rgb = self.resize_image(img_rgb)
        
        # Step 4: Enhancement (CLAHE on Value channel)
        if enhance:
            h, s, v = cv2.split(img_hsv)
            v_enhanced = self.enhance_image(v, method='clahe')
            img_hsv = cv2.merge([h, s, v_enhanced])
        
        # Step 5: Normalization (channel-wise)
        if normalize:
            h, s, v = cv2.split(img_hsv)
            # Normalize each channel with appropriate method
            h_norm = self.normalize_image(h, method='percentile')
            s_norm = self.normalize_image(s, method='percentile')
            v_norm = self.normalize_image(v, method='percentile')
            
            img_hsv = cv2.merge([h_norm, s_norm, v_norm])
        
        # Step 6: Create grayscale version for texture analysis
        # Convert RGB to grayscale using luminance weights
        img_gray = self.to_grayscale(img_rgb)
        
        # Apply enhancement to grayscale if requested
        if enhance:
            img_gray = self.enhance_image(img_gray, method='clahe')
        
        # Normalize grayscale if requested
        if normalize:
            img_gray = self.normalize_image(img_gray, method='percentile')
        
        return img_hsv, img_rgb, img_gray
        

    def get_preprocessing_stats(self, img_gray: np.ndarray) -> dict:
        """
        Get statistics from preprocessed grayscale image for analysis
        
        Returns:
            Dictionary with grayscale statistics
        """
        stats = {
            'grayscale': {
                'min': img_gray.min(), 'max': img_gray.max(), 
                'mean': img_gray.mean(), 'std': img_gray.std()
            }
        }
        
        return stats

    def preprocess_for_texture_analysis(self, image_path: Path, 
                                      crop_coords: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Preprocess image specifically for texture-based classification methods
        
        Args:
            image_path: Path to the image file
            crop_coords: Optional crop coordinates (x1, y1, x2, y2)
            
        Returns:
            Preprocessed grayscale image ready for texture analysis
        """
        # Use the existing pipeline but only return the grayscale image
        _, _, gray_image = self.preprocess_pipeline(
            image_path, 
            crop_coords=crop_coords, 
            enhance=True, 
            normalize=True
        )
        
        return gray_image

if __name__ == "__main__":
    # Initialize processor
    processor = ImageProcessor(target_size=(512, 512))
    
    # Define crop coordinates (your previous coordinates)
    crop_coords = (103, 150, 1323, 1720)  # (x1, y1, x2, y2)
    
    # Process image with full pipeline
    image_path = Path('./data/raw/fultonATJ-1-043.jpg')
    hsv_image, rgb_image, gray_image = processor.preprocess_pipeline(
        image_path, 
        crop_coords=crop_coords,
        enhance=True,
        normalize=True
    )
    
    # Get statistics
    stats = processor.get_preprocessing_stats(gray_image)
    
    print("=== PREPROCESSING PIPELINE RESULTS ===")
    print(f"Image shape - RGB: {rgb_image.shape}, Gray: {gray_image.shape}")
    
    print("\n=== GRAYSCALE STATISTICS ===")
    print(f"Grayscale: {stats['grayscale']['min']:.1f} - {stats['grayscale']['max']:.1f} (mean: {stats['grayscale']['mean']:.1f})")
    
    print("\n=== NEXT STEPS ===")
    print("Use the preprocessed grayscale image with baseline classification methods:")
    print("from src.baseline_methods.threshold_classification import ThresholdClassifier")
    print("classifier = ThresholdClassifier()")
    print("forest_mask, threshold = classifier.classify_forest(gray_image)")
    
    # Display preprocessing results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image (RGB)')
    axes[0].axis('off')
    
    # Grayscale image
    axes[1].imshow(gray_image, cmap='gray')
    axes[1].set_title('Preprocessed Grayscale\n(Ready for Classification)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()