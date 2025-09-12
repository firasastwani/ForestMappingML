# ForestMappingML

# AI@UGA Forest Mapping Project

A machine learning pipeline for forest vs. non-forest classification using historic aerial imagery from Fulton County, Georgia. This project implements multiple classification approaches including baseline thresholding methods, texture-based filtering, and a Random Forest classifier to map forest areas from 1938 aerial photography.

## Method Comparison Results

![Method Comparison](results/method_comparison.png)

_Comprehensive comparison of all classification methods applied to 1938 Fulton County aerial photography. The visualization shows original image, preprocessing steps, baseline thresholding methods, texture-based approaches, and machine learning results with forest percentages for each method._

## Project Overview

**Key Objectives:**

- Develop automated forest classification from historic aerial imagery
- Compare multiple classification approaches (thresholding, texture-based, ML)
- Create a scalable pipeline for city-wide forest mapping
- Provide methodology justification for each technical decision

## Dataset

### Source

**Georgia Aerial Photography Collection - Fulton County, 1938**

- Geographic Focus: Fulton County, Atlanta area (selected for local relevance)
- Scope: 1938 aerial photography index
- Image Specifications: 1526 x 1791 pixels per full image

### Data Selection Rationale

- **Consistent Sensor Data**: Initially focused on consistent gradient and size characteristics for better training results
- **Manageable ROI**: Selected small, clearly defined regions showing both forest and non-forested areas
- **Local Relevance**: I choose Fulton county because I live there, and it also represents Atlanta's urban-forest interface

### Sample Images

The dataset includes 5 sample images from different indices:

- `fultonATJ-1-043.jpg` - Primary training image
- `fultonATJ-2-009.jpg`, `fultonATJ-2-014.jpg`, `fultonATJ-2-016.jpg` - Test images
- `fultonATJ-3A-023.jpg` - Additional validation image

## Methodology

### Preprocessing Pipeline

The preprocessing pipeline was designed to enhance forest classification accuracy while preserving important texture information.

#### Color Space Conversion: RGB → HSV

**Decision**: Convert images from RGB to HSV color space
**Justification**:

- **Separation of Color and Intensity**: HSV separates color information from intensity, providing robustness to lighting variations
- **Simplified Segmentation**: Enables more effective color-based segmentation and thresholding
- **Aerial Imagery Optimization**: Better suited for aerial imagery analysis where lighting conditions vary across the image

#### Image Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Decision**: Use CLAHE over traditional histogram equalization or other enhancement methods
**Justification**:

**CLAHE Advantages:**

- **Local Contrast Enhancement**: Equalizes histograms in small regions (8x8 tiles) rather than globally
- **Noise Prevention**: Prevents over-amplification of noise common in aerial imagery
- **Detail Preservation**: Improves visibility of details in both dark and bright areas
- **Forest Texture Enhancement**: Makes forest textures more visible and enhances subtle differences between forest types

**Rejected Alternatives:**

- **Traditional Histogram Equalization**: Applies same transformation globally, can over-amplify noise and create artifacts
- **Gaussian Blur**: Smooths out important forest textures and reduces edge information crucial for forest boundaries
- **Traditional Sharpening**: Amplifies noise along with edges, creating false edges in aerial imagery

#### Normalization Strategy

**Decision**: Implement percentile-based normalization as default method
**Justification**:

- **Robustness**: Ignores extreme outliers (brightest/darkest pixels)
- **Artifact Handling**: Handles compression artifacts common in historic imagery
- **Consistency**: Ensures consistent input ranges for ML classifiers and baseline methods
- **Comparison Capability**: Enables fair comparison between different images with varying brightness/contrast

**Alternative Considered**: Mean-standard deviation normalization

- **Result**: Surprisingly close results to percentile method
- **Decision**: Chose mean-std as default due to simplicity while maintaining accuracy

### Baseline Methods

#### 1. Threshold Classification

**Approach**: Statistical thresholding using mean - standard deviation
**Threshold Calculation**: `threshold = image.mean() - image.std()`

**Rationale**:

- **Forest Characteristics**: Forest areas are typically darker due to:
  - Tree canopy shadows
  - Dense vegetation absorption
  - Natural darkness of vegetation
- **Non-Forest Characteristics**: Non-forest areas are brighter due to:
  - Open fields with less shadow coverage
  - Reflective surfaces (roads, buildings)
  - Less dense vegetation

**Method Comparison**:

- **Mean-Std**: `mean - std` - Simple, effective statistical approach
- **Percentile**: 25th percentile - More robust but computationally complex
- **Result**: Both methods produced nearly identical results, mean-std chosen for simplicity

#### 2. Texture-Based Filtering

**Approach**: Edge detection using Sobel and Canny operators
**Justification**: Forests exhibit different texture characteristics than non-forest areas

**Forest Texture Characteristics**:

- **Rough, irregular patterns** due to canopy structure
- **High edge density** from tree boundaries and foliage
- **Complex texture patterns** from overlapping vegetation

**Non-Forest Texture Characteristics**:

- **Smoother, uniform areas** (fields, water)
- **Sharp geometric edges** (buildings, roads)
- **Lower edge density** overall

**Implementation Details**:

**Sobel Edge Detection**:

- Computes gradient magnitude: `sqrt(sobelx² + sobely²)`
- Provides "raw texture density" measurement
- Uses 70th percentile threshold for forest classification

**Canny Edge Detection**:

- Advanced detector with noise smoothing
- Non-maximum suppression and hysteresis thresholding
- Produces cleaner, thinner edges
- Uses mean intensity as threshold

**Method Selection**: Both methods complement each other - Sobel for texture density, Canny for refined edge mapping.

### Machine Learning Approach

#### Random Forest Classifier

**Algorithm Choice**: Random Forest with 50 decision trees
**Justification**:

- **Robustness**: Handles noise well, important for historic aerial imagery
- **Speed**: Fast training and prediction suitable for this prototype
- **Feature Importance**: Provides insight into which features matter most
- **Overfitting Resistance**: Ensemble method reduces overfitting risk

#### Feature Engineering

**Extracted Features**:

1. **RGB Values**: Red, Green, Blue pixel intensities
2. **HSV Values**: Hue, Saturation, Value converted from RGB
3. **Feature Scaling**: StandardScaler for consistent feature ranges

**Training Process**:

1. **Manual Labeling**: Interactive sample collection for forest/non-forest pixels
2. **Feature Extraction**: Convert pixel coordinates to numerical features
3. **Model Training**: Random Forest learns patterns from labeled samples
4. **Image Classification**: Apply trained model to classify entire images

**Sample Collection**: 15-20 samples each for forest and non-forest classes

## Results & Analysis

### Performance Comparison

**Baseline Methods**:

- **Threshold Classification**: Effective for clear forest/non-forest distinction
- **Texture-Based Methods**: Better for complex boundary detection
- **Combined Approach**: Most robust for varied imagery

**Machine Learning**:

- **Superior Performance**: Random Forest outperformed baseline methods
- **Feature Learning**: Automatically learns optimal feature combinations
- **Generalization**: Better performance across different images

### Key Findings

1. **Preprocessing Impact**: CLAHE enhancement significantly improved classification accuracy
2. **Method Complementarity**: Different methods excel in different scenarios
3. **Feature Importance**: HSV features proved more discriminative than RGB alone
4. **Scalability**: Pipeline successfully processes multiple images consistently

### Error Handling

Comprehensive error handling implemented throughout:

- File existence validation
- Image loading error handling
- Feature extraction error management
- Model training validation
- Graceful degradation for missing images

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn
- Pathlib

## Future Improvements

### Scalability Improvements

1. **Batch Processing**: Optimize for large-scale city-wide processing
2. **Cloud Deployment**: Enable distributed processing capabilities
3. **Automated Labeling**: Reduce manual annotation requirements
4. **Integration**: Connect with GIS systems for practical deployment

### Data Expansion

1. **Multi-Sensor Integration**: Include different aerial photography sensors
2. **Geographic Expansion**: Extend beyond Fulton County
3. **Temporal Range**: Incorporate imagery from multiple decades
4. **Validation Data**: Add ground truth validation datasets

## Conclusion

This project successfully demonstrates a comprehensive approach to forest classification from historic aerial imagery. The combination of traditional computer vision techniques with modern machine learning provides both interpretability and performance. The methodology justification ensures that each technical decision contributes to the overall goal of accurate, scalable forest mapping for urban planning applications.

The pipeline serves as a solid foundation for the City of Atlanta's forest age mapping initiative, with clear paths for enhancement and scaling to city-wide deployment.
