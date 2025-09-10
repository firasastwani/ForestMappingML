"""
Image preprocessing utilities for forest maps
Handles loading, cropping, preprocessing, and normilization of arial imagery
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


image1 = cv2.imread('./data/raw/fultonIndex6.jpg')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.imshow(image1)
plt.show()