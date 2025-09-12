"""
Lightweight Pixel-based Forest Classification
Simple Random Forest for forest vs non-forest pixel classification
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class PixelClassifier:
    """Simple ML classifier for forest vs non-forest pixels"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, image, coords):
        """Extract simple features for pixel coordinates"""
        features = []
        for y, x in coords:
            # Basic RGB features
            if len(image.shape) == 3:
                r, g, b = image[y, x]
                # Convert to HSV
                hsv = cv2.cvtColor(image[y:y+1, x:x+1], cv2.COLOR_RGB2HSV)[0, 0]
                h, s, v = hsv
                features.append([r, g, b, h, s, v])
            else:
                # Grayscale
                gray = image[y, x]
                features.append([gray, gray, gray, 0, 0, gray])
        return np.array(features)
    
    def train(self, image, forest_coords, non_forest_coords):
        """Train the classifier with sample points"""
        # Extract features
        forest_features = self.extract_features(image, forest_coords)
        non_forest_features = self.extract_features(image, non_forest_coords)
        
        # Combine data
        X = np.vstack([forest_features, non_forest_features])
        y = np.hstack([np.ones(len(forest_coords)), np.zeros(len(non_forest_coords))])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Trained on {len(forest_coords)} forest + {len(non_forest_coords)} non-forest samples")
        return self.model.score(X_scaled, y)
    
    def classify_image(self, image):
        """Classify entire image"""
        
        h, w = image.shape[:2]
        coords = [(y, x) for y in range(h) for x in range(w)]
        
        # Extract features for all pixels
        features = self.extract_features(image, coords)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        return predictions.reshape(h, w)
    
    def classify_multiple_images(self, image_paths, crop_coords=None):
        """Classify multiple images using the trained model"""
        
        if not self.is_trained:
            raise ValueError("Train the model first!")
        
        from pathlib import Path
        from src.preprocessing.image_processer import ImageProcessor
        
        processor = ImageProcessor(target_size=(256, 256))
        results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"Classifying image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                # Load and preprocess image
                hsv_image, rgb_image, gray_image = processor.preprocess_pipeline(
                    Path(image_path), crop_coords=crop_coords
                )
                
                # Classify
                forest_mask = self.classify_image(rgb_image)
                forest_pct = (np.sum(forest_mask) / forest_mask.size) * 100
                
                results[Path(image_path).name] = {
                    'forest_mask': forest_mask,
                    'forest_percentage': forest_pct,
                    'rgb_image': rgb_image
                }
                
                print(f"  Forest: {forest_pct:.1f}%")
                
            except Exception as e:
                print(f"  Error processing {Path(image_path).name}: {e}")
                results[Path(image_path).name] = {'error': str(e)}
        
        return results
    
    def visualize_samples(self, image, forest_coords, non_forest_coords):
        """Show sample points on image"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        
        # Plot samples
        if forest_coords:
            forest_y, forest_x = zip(*forest_coords)
            plt.scatter(forest_x, forest_y, c='green', s=50, alpha=0.8, label=f'Forest ({len(forest_coords)})')
        
        if non_forest_coords:
            non_forest_y, non_forest_x = zip(*non_forest_coords)
            plt.scatter(non_forest_x, non_forest_y, c='red', s=50, alpha=0.8, label=f'Non-Forest ({len(non_forest_coords)})')
        
        plt.title('Training Samples')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self, image, forest_mask):
        """Show classification results"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(forest_mask, cmap='gray')
        forest_pct = (np.sum(forest_mask) / forest_mask.size) * 100
        plt.title(f'Forest Mask\n{forest_pct:.1f}% forest')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        overlay = image.copy()
        if len(overlay.shape) == 3:
            overlay[forest_mask == 1] = [0, 255, 0]  # Green for forest
        else:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
            overlay[forest_mask == 1] = [0, 255, 0]
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_multiple_results(self, results, max_images=4):
        """Visualize results from multiple images"""
        n_images = min(len(results), max_images)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i, (img_name, result) in enumerate(list(results.items())[:n_images]):
            if 'error' in result:
                continue
                
            # Original image
            axes[0, i].imshow(result['rgb_image'])
            axes[0, i].set_title(f'{img_name}\nOriginal')
            axes[0, i].axis('off')
            
            # Forest mask
            forest_pct = result['forest_percentage']
            axes[1, i].imshow(result['forest_mask'], cmap='gray')
            axes[1, i].set_title(f'Forest: {forest_pct:.1f}%')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


class SampleCollector:
    """Interactive sample collection"""
    
    def __init__(self, image):
        self.image = image
        self.forest_coords = []
        self.non_forest_coords = []
        self.collecting_forest = True
        
    def collect_samples(self, n_forest=20, n_non_forest=20):
        """Interactive sample collection"""
        print(f"Click on {n_forest} forest pixels (green areas)")
        print("Press 'f' to switch to non-forest, 'q' when done")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image)
        ax.set_title(f"Click on {n_forest} forest pixels")
        
        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
                
            y, x = int(event.ydata), int(event.xdata)
            
            if self.collecting_forest and len(self.forest_coords) < n_forest:
                self.forest_coords.append((y, x))
                ax.plot(x, y, 'go', markersize=8, alpha=0.7)
                remaining = n_forest - len(self.forest_coords)
                ax.set_title(f"Forest: {len(self.forest_coords)}/{n_forest} (remaining: {remaining})")
            elif not self.collecting_forest and len(self.non_forest_coords) < n_non_forest:
                self.non_forest_coords.append((y, x))
                ax.plot(x, y, 'ro', markersize=8, alpha=0.7)
                remaining = n_non_forest - len(self.non_forest_coords)
                ax.set_title(f"Non-forest: {len(self.non_forest_coords)}/{n_non_forest} (remaining: {remaining})")
            
            fig.canvas.draw()
        
        def on_key(event):
            if event.key == 'f':
                self.collecting_forest = False
                remaining = n_non_forest - len(self.non_forest_coords)
                ax.set_title(f"Click on {remaining} non-forest pixels")
                print(f"Switched to non-forest collection. Click on {remaining} non-forest pixels")
            elif event.key == 'q':
                plt.close(fig)
                print(f"Done! Forest: {len(self.forest_coords)}, Non-forest: {len(self.non_forest_coords)}")
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        return self.forest_coords, self.non_forest_coords


# Simple usage example
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src.preprocessing.image_processer import ImageProcessor
        
        print("=== LIGHTWEIGHT ML CLASSIFICATION ===")
        
        # Load first image for training
        processor = ImageProcessor(target_size=(256, 256))  # Smaller for speed
        train_image_path = Path('./data/raw/fultonATJ-1-043.jpg')
        crop_coords = (103, 150, 1323, 1720)
        
        hsv_image, rgb_image, gray_image = processor.preprocess_pipeline(
            train_image_path, crop_coords=crop_coords
        )
        
        print("1. Training on first image...")
        
        # Interactive sample collection
        collector = SampleCollector(rgb_image)
        forest_coords, non_forest_coords = collector.collect_samples(n_forest=15, n_non_forest=15)
        
        if len(forest_coords) > 0 and len(non_forest_coords) > 0:
            # Train classifier
            classifier = PixelClassifier()
            accuracy = classifier.train(rgb_image, forest_coords, non_forest_coords)
            print(f"Training accuracy: {accuracy:.3f}")
            
            # Show training results
            print("\n2. Training image results:")
            forest_mask = classifier.classify_image(rgb_image)
            classifier.visualize_results(rgb_image, forest_mask)
            
            # Classify other images
            print("\n3. Classifying other images...")
            other_images = [
                './data/raw/fultonATJ-2-009.jpg',
                './data/raw/fultonATJ-2-014.jpg',
                './data/raw/fultonATJ-2-016.jpg',
                './data/raw/fultonATJ-3A-023.jpg'
            ]
            
            # Filter to only existing images
            existing_images = [img for img in other_images if Path(img).exists()]
            
            if existing_images:
                results = classifier.classify_multiple_images(existing_images, crop_coords)
                
                # Show results
                print("\n4. Results summary:")
                for img_name, result in results.items():
                    if 'error' not in result:
                        print(f"  {img_name}: {result['forest_percentage']:.1f}% forest")
                
                # Visualize results
                print("\n5. Visualizing results...")
                classifier.visualize_multiple_results(results)
            else:
                print("No other images found to classify")
        else:
            print("Need both forest and non-forest samples!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()