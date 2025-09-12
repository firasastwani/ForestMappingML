"""
Threshold-based forest classification baseline method
Implements simple thresholding for forest vs non-forest detection
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class ThresholdClassifier:
    """
    Baseline forest classification using thresholding method
    """
    
    def __init__(self):
        self.threshold_methods = ['mean_std', 'percentile']
    
    def calculate_threshold(self, image: np.ndarray, method: str = 'mean_std') -> float:
        """
        Calculate optimal threshold for forest classification
        
        Args:
            image: Grayscale image
            method: Threshold calculation method
            
        Returns:
            Threshold value
        """
        if method == 'mean_std':
            # Statistical threshold: mean - 1 standard deviation
            return image.mean() - image.std()

        elif method == 'percentile':
            # Percentile-based threshold (25th percentile)
            return np.percentile(image, 25)
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
    
    def classify_forest(self, image: np.ndarray, threshold: Optional[float] = None, 
                       method: str = 'mean_std') -> Tuple[np.ndarray, float]:
        """
        Classify pixels as forest or non-forest using thresholding
        
        Args:
            image: Grayscale image
            threshold: Manual threshold (if None, will calculate automatically)
            method: Method for automatic threshold calculation
            
        Returns:
            Tuple of (binary_mask, threshold_used)
            - binary_mask: True = forest, False = non-forest
            - threshold_used: The threshold value used
        """
        if threshold is None:
            threshold = self.calculate_threshold(image, method)
        
        # Create binary mask: forest = darker pixels (below threshold)
        forest_mask = image < threshold
        
        return forest_mask, threshold
    
    def evaluate_threshold_methods(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate different threshold methods and return statistics
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with results for each threshold method
        """
        results = {}
        
        for method in self.threshold_methods:
            threshold = self.calculate_threshold(image, method)
            forest_mask, _ = self.classify_forest(image, threshold)
            
            # Calculate statistics
            total_pixels = image.size
            forest_pixels = np.sum(forest_mask)
            forest_percentage = (forest_pixels / total_pixels) * 100
            
            results[method] = {
                'threshold': threshold,
                'forest_pixels': forest_pixels,
                'forest_percentage': forest_percentage,
                'non_forest_pixels': total_pixels - forest_pixels,
                'non_forest_percentage': 100 - forest_percentage
            }
        
        return results
    
    def visualize_results(self, image: np.ndarray, forest_mask: np.ndarray, 
                         threshold: float, method: str = 'threshold') -> None:
        """
        Visualize threshold classification results
        
        Args:
            image: Original grayscale image
            forest_mask: Binary forest mask
            threshold: Threshold value used
            method: Method name for title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')
        
        # Forest mask
        axes[0, 1].imshow(forest_mask, cmap='gray')
        axes[0, 1].set_title(f'Forest Mask ({method})\nWhite = Forest, Black = Non-Forest')
        axes[0, 1].axis('off')
        
        # Histogram with threshold
        axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color='gray')
        axes[1, 0].axvline(threshold, color='red', linestyle='--', 
                          label=f'Threshold: {threshold:.1f}')
        axes[1, 0].set_title('Pixel Intensity Distribution')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Overlay visualization
        overlay = image.copy()
        overlay[forest_mask] = overlay[forest_mask] * 0.5  # Darken forest areas
        axes[1, 1].imshow(overlay, cmap='gray')
        axes[1, 1].set_title('Overlay: Forest Areas Highlighted')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    def compare_methods(self, image: np.ndarray) -> None:
        """
        Compare mean_std and percentile methods side by side
        
        Args:
            image: Grayscale image
        """
        results = self.evaluate_threshold_methods(image)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Mean-Std method
        mean_std_result = results['mean_std']
        mean_std_mask, _ = self.classify_forest(image, mean_std_result['threshold'])
        axes[0, 1].imshow(mean_std_mask, cmap='gray')
        axes[0, 1].set_title(f'Mean-Std Method\n'
                           f'Threshold: {mean_std_result["threshold"]:.1f}\n'
                           f'Forest: {mean_std_result["forest_percentage"]:.1f}%')
        axes[0, 1].axis('off')
        
        # Percentile method
        percentile_result = results['percentile']
        percentile_mask, _ = self.classify_forest(image, percentile_result['threshold'])
        axes[1, 0].imshow(percentile_mask, cmap='gray')
        axes[1, 0].set_title(f'Percentile Method\n'
                           f'Threshold: {percentile_result["threshold"]:.1f}\n'
                           f'Forest: {percentile_result["forest_percentage"]:.1f}%')
        axes[1, 0].axis('off')
        
        # Method comparison text
        axes[1, 1].axis('off')
        summary_text = "Method Comparison:\n\n"
        summary_text += f"Mean-Std:\n"
        summary_text += f"  Threshold: {mean_std_result['threshold']:.1f}\n"
        summary_text += f"  Forest: {mean_std_result['forest_percentage']:.1f}%\n"
        summary_text += f"  Non-forest: {mean_std_result['non_forest_percentage']:.1f}%\n\n"
        summary_text += f"Percentile:\n"
        summary_text += f"  Threshold: {percentile_result['threshold']:.1f}\n"
        summary_text += f"  Forest: {percentile_result['forest_percentage']:.1f}%\n"
        summary_text += f"  Non-forest: {percentile_result['non_forest_percentage']:.1f}%\n\n"
        summary_text += f"Difference:\n"
        summary_text += f"  Forest %: {abs(mean_std_result['forest_percentage'] - percentile_result['forest_percentage']):.1f}%"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    

# Basic testing and demonstration
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src.preprocessing.image_processer import ImageProcessor
        
        # Initialize components
        processor = ImageProcessor(target_size=(512, 512))
        classifier = ThresholdClassifier()
        
        # Test with first available image
        test_images = [
            './data/raw/fultonATJ-1-043.jpg',
            './data/raw/fultonATJ-2-009.jpg',
            './data/raw/fultonATJ-2-014.jpg',
            './data/raw/fultonATJ-2-016.jpg',
            './data/raw/fultonATJ-3A-023.jpg'
        ]
        
        image_path = None
        for img_path in test_images:
            p = Path(img_path)
            if p.exists():
                image_path = p
                break
        
        if image_path is None:
            print("No demo image found. Please add a raw image to data/raw.")
            sys.exit(0)
        
        # Standard crop
        crop_coords = (103, 150, 1323, 1720)
        
        # Preprocess image
        _, _, gray_image = processor.preprocess_pipeline(
            image_path, crop_coords=crop_coords
        )
        
        # Test threshold methods
        results = classifier.evaluate_threshold_methods(gray_image)
        
        # Print results
        print("Threshold Method Comparison:")
        for method, result in results.items():
            print(f"{method}: {result['forest_percentage']:.1f}% forest (threshold: {result['threshold']:.1f})")
        
        # Test individual classification
        forest_mask, threshold = classifier.classify_forest(gray_image, method='mean_std')
        forest_pct = (np.sum(forest_mask) / gray_image.size) * 100
        print(f"Mean-Std Method: {forest_pct:.1f}% forest")
        
        # Visualize results
        classifier.visualize_results(gray_image, forest_mask, threshold, 'Mean-Std')
        classifier.compare_methods(gray_image)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check that image files exist and paths are correct")
