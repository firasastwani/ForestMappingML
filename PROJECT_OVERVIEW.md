# AI@UGA Forest Mapping Project - Requirements Overview

## Project Objective

Develop a prototype pipeline that takes historic aerial imagery and produces a forest vs. non-forest map for one or two years of data, as part of a collaboration with the City of Atlanta to map forest age.

## Core Requirements

### Technical Stack

- **Language**: Python
- **Allowed Libraries**: OpenCV, Rasterio, scikit-image, scikit-learn, PyTorch/TensorFlow
- **Development Environment**: Jupyter Notebook or Python script
- **Scope**: ~1 week completion time

### Dataset Requirements

- **Source Options**:
  - UGA Aerial Photography Collection (preferred)
  - USGS Earth Explorer
  - Alternative aerial/satellite imagery if download is difficult
- **Geographic Focus**: Atlanta area
- **Temporal Scope**: 1-2 years of data
- **Processing**: Small region of interest (ROI) selection

### Technical Implementation Requirements

#### 1. Preprocessing Pipeline

- Convert image to grayscale
- Resize or crop for easier processing
- Apply filters: blurring, sharpening, histogram equalization
- Document preprocessing choices and rationale

#### 2. Baseline Forest Classification Methods (Minimum 2 Required)

- **Thresholding**: Classify pixels above/below intensity threshold
- **Texture-based filtering**: Use edge detection (Sobel, Canny) or local variance
- **Clustering**: Use k-means on pixel intensities
- **Output**: Binary masks (forest = white, non-forest = black)
- **Visualization**: Side-by-side comparison of methods

#### 3. Machine Learning Classifier

- **Data Collection**: Manual labeling of pixel samples for "forest" and "non-forest"
- **Classifier Options**: Logistic regression, decision tree, or random forest
- **Training**: Train classifier on labeled samples
- **Application**: Apply to classify entire image
- **Evaluation**: Compare ML results with baseline methods

#### 4. Results and Analysis

- **Visual Comparisons**: Show results from all implemented methods
- **Performance Analysis**: Describe and compare method effectiveness
- **Methodology Justification**: Explain any deviations from suggested approach

### Deliverables (Required)

#### 1. Code Implementation

- **Format**: Jupyter Notebook or Python script
- **Structure**: Well-organized, documented code
- **Functionality**: Complete pipeline from data loading to results

#### 2. Documentation

- **README.md**: Comprehensive project description including:
  - Methodology explanation
  - Approach justification
  - Results summary
  - Setup instructions
  - Dependencies list

#### 3. Demo Video

- **Duration**: Brief demonstration of project
- **Content**: Walkthrough of approach and results
- **Purpose**: Showcase implementation and analysis

### Evaluation Criteria

#### Technical Competency

- **Computer Vision**: Proper use of image processing techniques
- **Machine Learning**: Appropriate algorithm selection and implementation
- **Geospatial Analysis**: Understanding of aerial imagery characteristics
- **Code Quality**: Clean, efficient, well-documented implementation

#### Problem-Solving Approach

- **Methodology**: Logical progression from simple to complex methods
- **Innovation**: Creative solutions or improvements to baseline approaches
- **Analysis**: Thoughtful evaluation of results and limitations
- **Documentation**: Clear explanation of decisions and trade-offs

#### Professional Presentation

- **Organization**: Structured, easy-to-follow implementation
- **Visualization**: Clear, informative plots and comparisons
- **Communication**: Professional documentation and video presentation

### Success Metrics

#### Minimum Viable Product

- [ ] At least 2 baseline classification methods implemented
- [ ] 1 machine learning classifier trained and applied
- [ ] Visual comparison of all methods
- [ ] Working pipeline from data input to results output

#### Enhanced Deliverables

- [ ] Multiple preprocessing techniques tested
- [ ] Quantitative performance metrics (accuracy, precision, recall)
- [ ] Error analysis and failure case discussion
- [ ] Scalability considerations for city-wide application
- [ ] Temporal analysis (if using 2 years of data)

#### Professional Standards

- [ ] Clean, modular code structure
- [ ] Comprehensive documentation
- [ ] Professional visualizations with proper labeling
- [ ] Clear methodology justification
- [ ] Future improvement suggestions

### Key Success Factors

#### Technical Excellence

- **Feature Engineering**: Beyond simple pixel intensity (texture, spatial features)
- **Validation**: Proper train/test split and performance evaluation
- **Preprocessing**: Smart enhancement techniques for better classification
- **Post-processing**: Morphological operations to clean up results

#### Domain Knowledge Application

- **Forest Characteristics**: Understanding that forests are darker, more textured, irregular boundaries
- **Aerial Imagery**: Knowledge of typical aerial photography characteristics
- **Geospatial Context**: Consideration of Atlanta's urban-forest interface

#### Presentation Quality

- **Visual Consistency**: Professional color schemes and labeling
- **Method Comparison**: Clear side-by-side visualizations
- **Documentation**: README that enables reproduction of results
- **Video Demo**: Engaging walkthrough of methodology and results

### Timeline Considerations

- **Day 1**: Setup, data acquisition, initial exploration
- **Day 2**: Preprocessing pipeline, baseline methods implementation
- **Day 3**: Machine learning classifier development and training
- **Day 4**: Results analysis, documentation, video creation

### Risk Mitigation

- **Data Issues**: Have backup data sources ready
- **Technical Complexity**: Start simple, enhance iteratively
- **Time Management**: Prioritize working pipeline over perfect individual components
- **Scope Creep**: Focus on core requirements first, add enhancements if time permits

---

_This overview serves as a comprehensive reference for the AI@UGA Forest Mapping Project, ensuring all requirements are met while maintaining focus on delivering a high-quality, professional submission._
