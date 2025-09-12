"""
Texture-based forest classification baseline method
Implements Sobel and Canny edge detection for forest vs non-forest detection
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class TextureClassifier:
    """
    Baseline forest classification using texture-based methods
    """
    
    def __init__(self):
        self.texture_methods = ['sobel', 'canny']
        self.processor = None  # Will be set when needed
    
    def detect_edges(self, image: np.ndarray, method: str = 'sobel') -> np.ndarray:
        """
        Detect edges using specified method
        
        Args:
            image: Grayscale image
            method: 'sobel' or 'canny'
            
        Returns:
            Edge detection result
        """
        if method == 'sobel':
            # Sobel edge detection
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            return magnitude.astype(np.uint8)
            
        elif method == 'canny':
            # Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            return edges
            
        else:
            raise ValueError(f"Unknown texture method: {method}")
    
    def classify_forest(self, image: np.ndarray, method: str = 'sobel', 
                       threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Classify pixels as forest or non-forest using texture analysis
        
        Args:
            image: Grayscale image
            method: 'sobel' or 'canny'
            threshold: Manual threshold (if None, will calculate automatically)
            
        Returns:
            Tuple of (binary_mask, threshold_used)
            - binary_mask: True = forest (high texture), False = non-forest (low texture)
            - threshold_used: The threshold value used
        """
        # Detect edges
        edges = self.detect_edges(image, method)
        
        if threshold is None:
            if method == 'sobel':
                # Use percentile threshold for Sobel magnitude
                threshold = np.percentile(edges, 70)
            elif method == 'canny':
                # Canny already produces binary, use mean as threshold
                threshold = edges.mean()
        
        # Create binary mask: forest = high texture (above threshold)
        forest_mask = edges > threshold
        
        return forest_mask, threshold
    
    def evaluate_texture_methods(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate different texture methods and return statistics
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with results for each texture method
        """
        results = {}
        
        for method in self.texture_methods:
            forest_mask, threshold = self.classify_forest(image, method)
            
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
                         edges: np.ndarray, threshold: float, method: str = 'texture') -> None:
        """
        Visualize texture classification results
        
        Args:
            image: Original grayscale image
            forest_mask: Binary forest mask
            edges: Edge detection result
            threshold: Threshold value used
            method: Method name for title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')
        
        # Edge detection result
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title(f'{method.title()} Edge Detection\nThreshold: {threshold:.1f}')
        axes[0, 1].axis('off')
        
        # Forest mask
        axes[1, 0].imshow(forest_mask, cmap='gray')
        axes[1, 0].set_title(f'Forest Mask ({method})\nWhite = Forest, Black = Non-Forest')
        axes[1, 0].axis('off')
        
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
        Compare Sobel and Canny methods side by side
        
        Args:
            image: Grayscale image
        """
        results = self.evaluate_texture_methods(image)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Sobel method
        sobel_edges = self.detect_edges(image, 'sobel')
        sobel_result = results['sobel']
        sobel_mask, _ = self.classify_forest(image, 'sobel', sobel_result['threshold'])
        
        axes[0, 1].imshow(sobel_edges, cmap='gray')
        axes[0, 1].set_title(f'Sobel Edge Detection\nThreshold: {sobel_result["threshold"]:.1f}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sobel_mask, cmap='gray')
        axes[0, 2].set_title(f'Sobel Forest Mask\nForest: {sobel_result["forest_percentage"]:.1f}%')
        axes[0, 2].axis('off')
        
        # Canny method
        canny_edges = self.detect_edges(image, 'canny')
        canny_result = results['canny']
        canny_mask, _ = self.classify_forest(image, 'canny', canny_result['threshold'])
        
        axes[1, 0].imshow(canny_edges, cmap='gray')
        axes[1, 0].set_title(f'Canny Edge Detection\nThreshold: {canny_result["threshold"]:.1f}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(canny_mask, cmap='gray')
        axes[1, 1].set_title(f'Canny Forest Mask\nForest: {canny_result["forest_percentage"]:.1f}%')
        axes[1, 1].axis('off')
        
        # Method comparison text
        axes[1, 2].axis('off')
        summary_text = "Method Comparison:\n\n"
        summary_text += f"Sobel:\n"
        summary_text += f"  Threshold: {sobel_result['threshold']:.1f}\n"
        summary_text += f"  Forest: {sobel_result['forest_percentage']:.1f}%\n"
        summary_text += f"  Non-forest: {sobel_result['non_forest_percentage']:.1f}%\n\n"
        summary_text += f"Canny:\n"
        summary_text += f"  Threshold: {canny_result['threshold']:.1f}\n"
        summary_text += f"  Forest: {canny_result['forest_percentage']:.1f}%\n"
        summary_text += f"  Non-forest: {canny_result['non_forest_percentage']:.1f}%\n\n"
        summary_text += f"Difference:\n"
        summary_text += f"  Forest %: {abs(sobel_result['forest_percentage'] - canny_result['forest_percentage']):.1f}%"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

    def classify_from_path(self, image_path, crop_coords: Optional[Tuple[int, int, int, int]] = None, 
                          method: str = 'sobel') -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Complete pipeline: preprocess image and classify forest areas
        
        Args:
            image_path: Path to the image file
            crop_coords: Optional crop coordinates (x1, y1, x2, y2)
            method: 'sobel' or 'canny'
            
        Returns:
            Tuple of (forest_mask, threshold, preprocessed_image)
        """
        from pathlib import Path
        from src.preprocessing.image_processer import ImageProcessor
        
        # Initialize processor if not already done
        if self.processor is None:
            self.processor = ImageProcessor(target_size=(512, 512))
        
        # Preprocess image for texture analysis
        gray_image = self.processor.preprocess_for_texture_analysis(
            Path(image_path), crop_coords=crop_coords
        )
        
        # Classify forest areas
        forest_mask, threshold = self.classify_forest(gray_image, method)
        
        return forest_mask, threshold, gray_image


# Basic testing and demonstration
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src.preprocessing.image_processer import ImageProcessor
        
        # Initialize classifier (it will handle its own processor)
        classifier = TextureClassifier()
        
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
        
        # Test Sobel method using the complete pipeline
        sobel_mask, sobel_threshold, gray_image = classifier.classify_from_path(
            image_path, crop_coords=crop_coords, method='sobel'
        )
        sobel_edges = classifier.detect_edges(gray_image, 'sobel')
        
        # Test Canny method using the complete pipeline
        canny_mask, canny_threshold, _ = classifier.classify_from_path(
            image_path, crop_coords=crop_coords, method='canny'
        )
        canny_edges = classifier.detect_edges(gray_image, 'canny')
        
        # Visualize Sobel results
        classifier.visualize_results(gray_image, sobel_mask, sobel_edges, sobel_threshold, 'Sobel')
        
        # Visualize Canny results
        classifier.visualize_results(gray_image, canny_mask, canny_edges, canny_threshold, 'Canny')
        
        # Compare both methods
        classifier.compare_methods(gray_image)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        
    except Exception as e:
        print(f" Unexpected error: {e}")
        print("Check that image files exist and paths are correct")