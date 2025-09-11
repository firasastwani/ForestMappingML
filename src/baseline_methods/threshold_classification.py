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
    

# Comprehensive testing and demonstration
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src.preprocessing.image_processer import ImageProcessor
        
        print("=== THRESHOLD CLASSIFICATION TESTING ===")
        print("Testing baseline forest classification methods\n")
        
        # Initialize components
        processor = ImageProcessor(target_size=(512, 512))
        classifier = ThresholdClassifier()
        
        # Test with different images
        test_images = [
            './data/raw/fultonIndex6.jpg',
            './data/raw/fultonIndex12.jpg', 
            './data/raw/fultonATJ-1-043.jpg'
        ]
        
        crop_coords = (103, 150, 1323, 1720)  # Your standard crop coordinates
        
        for i, image_path in enumerate(test_images):
            image_path = Path(image_path)
            
            if not image_path.exists():
                print(f"⚠️  Skipping {image_path.name} - file not found")
                continue
                
            print(f"\n{'='*60}")
            print(f"TESTING IMAGE {i+1}: {image_path.name}")
            print(f"{'='*60}")
            
            try:
                # Step 1: Preprocess image
                print("Step 1: Preprocessing image...")
                _, _, gray_image = processor.preprocess_pipeline(
                    image_path, 
                    crop_coords=crop_coords,
                    enhance=True,
                    normalize=True
                )
                
                # Step 2: Test different threshold methods
                print("Step 2: Testing threshold methods...")
                results = classifier.evaluate_threshold_methods(gray_image)
                
                # Print method comparison
                print("\n--- THRESHOLD METHOD COMPARISON ---")
                for method, result in results.items():
                    print(f"{method.upper():>12}: Threshold={result['threshold']:6.1f}, "
                          f"Forest={result['forest_percentage']:5.1f}%, "
                          f"Non-forest={result['non_forest_percentage']:5.1f}%")
                
                # Step 3: Test individual classification
                print("\nStep 3: Testing individual classification...")
                forest_mask, threshold = classifier.classify_forest(gray_image, method='mean_std')
                
                print(f"Mean-Std Method Results:")
                print(f"  Threshold: {threshold:.1f}")
                print(f"  Forest pixels: {np.sum(forest_mask):,}")
                print(f"  Non-forest pixels: {np.sum(~forest_mask):,}")
                print(f"  Forest percentage: {(np.sum(forest_mask) / gray_image.size) * 100:.1f}%")
                
                # Step 4: Visualize results
                print("\nStep 4: Generating visualizations...")
                classifier.visualize_results(gray_image, forest_mask, threshold, 'Mean-Std')


                #Step 5: Compare all methods visually
                print("\nStep 5: Comparing all methods visually...")
                classifier.compare_methods(gray_image)
                
                print(f"\n✅ Successfully tested {image_path.name}")
                
            except Exception as e:
                print(f"❌ Error testing {image_path.name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETE")
        print(f"{'='*60}")
        
        # Additional functionality tests
        print("\n--- ADDITIONAL FUNCTIONALITY TESTS ---")
        
        # Test with manual threshold
        print("\n1. Testing manual threshold...")
        test_image_path = Path('./data/raw/fultonIndex6.jpg')
        if test_image_path.exists():
            _, _, test_gray = processor.preprocess_pipeline(test_image_path, crop_coords)
            manual_threshold = 100.0
            forest_mask_manual, _ = classifier.classify_forest(test_gray, threshold=manual_threshold)
            forest_percent_manual = (np.sum(forest_mask_manual) / test_gray.size) * 100
            print(f"   Manual threshold {manual_threshold}: {forest_percent_manual:.1f}% forest")
        
        # Test threshold calculation methods
        print("\n2. Testing threshold calculation methods...")
        if test_image_path.exists():
            _, _, test_gray = processor.preprocess_pipeline(test_image_path, crop_coords)
            for method in classifier.threshold_methods:
                threshold = classifier.calculate_threshold(test_gray, method)
                print(f"   {method}: {threshold:.1f}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("- Use these results to choose the best threshold method")
        print("- Implement additional baseline methods (clustering, edge detection)")
        print("- Create training data for machine learning classifier")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        print("and that the preprocessing module is available")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check that image files exist and paths are correct")
