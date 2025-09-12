"""
AI@UGA Forest Mapping Project - Main Visualization Script
Demonstrates all classification methods side by side for project showcase
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.preprocessing.image_processer import ImageProcessor
from src.baseline_methods.threshold_classification import ThresholdClassifier
from src.baseline_methods.texture_based_filtering import TextureClassifier
from src.machine_learning.pixel_classifier import PixelClassifier, SampleCollector


def calculate_forest_percentage(mask):
    """Calculate percentage of forest pixels in a binary mask"""
    return (np.sum(mask) / mask.size) * 100


def visualize_all_methods():
    """Main visualization function showing all classification methods"""
    
    print("=== AI@UGA Forest Mapping Project - Method Comparison ===")
    print("Loading and preprocessing image...")
    
    # Initialize processor
    processor = ImageProcessor(target_size=(512, 512))
    
    # Use the primary training image for consistent comparison
    image_path = Path('./data/raw/fultonATJ-1-043.jpg')
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please ensure the image file exists in data/raw/")
        return
    
    # Standard crop coordinates for consistent comparison
    crop_coords = (103, 150, 1323, 1720)
    
    # Preprocess image
    hsv_image, rgb_image, gray_image = processor.preprocess_pipeline(
        image_path, 
        crop_coords=crop_coords,
        enhance=True,
        normalize=True
    )
    
    print("✓ Image preprocessed successfully")
    print("Running all classification methods...")
    
    # Initialize all classifiers
    threshold_classifier = ThresholdClassifier()
    texture_classifier = TextureClassifier()
    
    # Method 1: Threshold Classification (Mean-Std)
    print("- Running threshold classification (mean-std)...")
    threshold_mask, threshold_value = threshold_classifier.classify_forest(
        gray_image, method='mean_std'
    )
    threshold_pct = calculate_forest_percentage(threshold_mask)
    
    # Method 2: Threshold Classification (Percentile)
    print("- Running threshold classification (percentile)...")
    percentile_mask, percentile_value = threshold_classifier.classify_forest(
        gray_image, method='percentile'
    )
    percentile_pct = calculate_forest_percentage(percentile_mask)
    
    # Method 3: Texture Classification (Sobel)
    print("- Running texture classification (Sobel)...")
    sobel_mask, sobel_threshold = texture_classifier.classify_forest(
        gray_image, method='sobel'
    )
    sobel_pct = calculate_forest_percentage(sobel_mask)
    
    # Method 4: Texture Classification (Canny)
    print("- Running texture classification (Canny)...")
    canny_mask, canny_threshold = texture_classifier.classify_forest(
        gray_image, method='canny'
    )
    canny_pct = calculate_forest_percentage(canny_mask)
    
    # Method 5: Machine Learning (Random Forest)
    print("- Setting up machine learning classification...")
    print("  Interactive sample collection for ML training")
    
    # Interactive sample collection
    collector = SampleCollector(rgb_image)
    print("\n" + "="*50)
    print("INTERACTIVE SAMPLE COLLECTION")
    print("="*50)
    print("Instructions:")
    print("1. Click on 15 forest pixels (green/dark areas)")
    print("2. Press 'f' to switch to non-forest collection")
    print("3. Click on 15 non-forest pixels (bright/clear areas)")
    print("4. Press 'q' when done")
    print("="*50)
    
    forest_coords, non_forest_coords = collector.collect_samples(n_forest=15, n_non_forest=15)
    
    if len(forest_coords) == 0 or len(non_forest_coords) == 0:
        print("Error: Need both forest and non-forest samples!")
        print("Using default sample points for demonstration...")
        
        # Fallback to default points if no samples collected
        forest_coords = [
            (100, 100), (120, 150), (140, 200), (160, 250), (180, 300),
            (200, 100), (220, 150), (240, 200), (260, 250), (280, 300),
            (300, 100), (320, 150), (340, 200), (360, 250), (380, 300)
        ]
        
        non_forest_coords = [
            (50, 50), (70, 100), (90, 150), (110, 200), (130, 250),
            (250, 50), (270, 100), (290, 150), (310, 200), (330, 250),
            (400, 100), (420, 150), (440, 200), (460, 250), (480, 300)
        ]
    
    print(f"\n✓ Collected {len(forest_coords)} forest and {len(non_forest_coords)} non-forest samples")
    
    ml_classifier = PixelClassifier()
    training_accuracy = ml_classifier.train(rgb_image, forest_coords, non_forest_coords)
    ml_mask = ml_classifier.classify_image(rgb_image)
    ml_pct = calculate_forest_percentage(ml_mask)
    
    print(f"✓ ML Training Accuracy: {training_accuracy:.3f}")
    print("✓ All methods completed successfully")
    
    # Create comprehensive visualization
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
 
    
    # Row 1: Original and Preprocessing
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original Image\n(RGB)', fontweight='bold', fontsize=12, pad=20)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Preprocessed\n(Grayscale + CLAHE)', fontweight='bold', fontsize=12, pad=20)
    axes[0, 1].axis('off')
    
    # Show preprocessing info
    axes[0, 2].axis('off')
    preprocessing_text = "Preprocessing Pipeline:\n\n"
    preprocessing_text += "• RGB → HSV conversion\n"
    preprocessing_text += "• CLAHE enhancement\n"
    preprocessing_text += "• Percentile normalization\n"
    preprocessing_text += f"• Crop: {crop_coords}\n"
    preprocessing_text += f"• Size: {gray_image.shape}"
    
    axes[0, 2].text(0.05, 0.95, preprocessing_text, transform=axes[0, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))
    
    # Row 2: Baseline Methods
    axes[1, 0].imshow(threshold_mask, cmap='gray')
    axes[1, 0].set_title(f'Threshold (Mean-Std)\nForest: {threshold_pct:.1f}%\nThreshold: {threshold_value:.1f}', 
                        fontweight='bold', fontsize=11, pad=15)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(percentile_mask, cmap='gray')
    axes[1, 1].set_title(f'Threshold (Percentile)\nForest: {percentile_pct:.1f}%\nThreshold: {percentile_value:.1f}', 
                        fontweight='bold', fontsize=11, pad=15)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sobel_mask, cmap='gray')
    axes[1, 2].set_title(f'Texture (Sobel)\nForest: {sobel_pct:.1f}%\nThreshold: {sobel_threshold:.1f}', 
                        fontweight='bold', fontsize=11, pad=15)
    axes[1, 2].axis('off')
    
    # Row 3: Advanced Methods
    axes[2, 0].imshow(canny_mask, cmap='gray')
    axes[2, 0].set_title(f'Texture (Canny)\nForest: {canny_pct:.1f}%\nThreshold: {canny_threshold:.1f}', 
                        fontweight='bold', fontsize=11, pad=15)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ml_mask, cmap='gray')
    axes[2, 1].set_title(f'Random Forest (ML)\nForest: {ml_pct:.1f}%\nAccuracy: {training_accuracy:.3f}', 
                        fontweight='bold', fontsize=11, pad=15)
    axes[2, 1].axis('off')
    
    # Summary comparison
    axes[2, 2].axis('off')
    summary_text = "Method Comparison Summary:\n\n"
    summary_text += f"Threshold (Mean-Std): {threshold_pct:.1f}%\n"
    summary_text += f"Threshold (Percentile): {percentile_pct:.1f}%\n"
    summary_text += f"Texture (Sobel): {sobel_pct:.1f}%\n"
    summary_text += f"Texture (Canny): {canny_pct:.1f}%\n"
    summary_text += f"Random Forest (ML): {ml_pct:.1f}%\n\n"
    
    # Calculate range
    percentages = [threshold_pct, percentile_pct, sobel_pct, canny_pct, ml_pct]
    summary_text += f"Range: {min(percentages):.1f}% - {max(percentages):.1f}%\n"
    summary_text += f"Std Dev: {np.std(percentages):.1f}%\n\n"
    
    # Best performing method
    best_method = ["Mean-Std", "Percentile", "Sobel", "Canny", "Random Forest"][np.argmax(percentages)]
    summary_text += f"Highest: {best_method}\n({max(percentages):.1f}%)"
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    # Improved spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.1)
    
    # Save the visualization
    output_path = Path('./results/method_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Visualization saved to: {output_path}")
    print("\n=== METHOD COMPARISON RESULTS ===")
    print(f"Threshold (Mean-Std):     {threshold_pct:.1f}% forest")
    print(f"Threshold (Percentile):   {percentile_pct:.1f}% forest")
    print(f"Texture (Sobel):          {sobel_pct:.1f}% forest")
    print(f"Texture (Canny):          {canny_pct:.1f}% forest")
    print(f"Random Forest (ML):       {ml_pct:.1f}% forest")
    print(f"Range:                    {min(percentages):.1f}% - {max(percentages):.1f}%")
    print(f"Standard Deviation:       {np.std(percentages):.1f}%")
    
    plt.show()
    
    return {
        'threshold_mean_std': threshold_pct,
        'threshold_percentile': percentile_pct,
        'texture_sobel': sobel_pct,
        'texture_canny': canny_pct,
        'random_forest': ml_pct,
        'training_accuracy': training_accuracy
    }


if __name__ == "__main__":
    try:
        results = visualize_all_methods()
        print("\n✓ All methods completed successfully!")
        print("Use this visualization to showcase the project's comprehensive approach to forest classification.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure all required images are in data/raw/")
        print("2. Check that all dependencies are installed")
        print("3. Verify the project structure is correct")
