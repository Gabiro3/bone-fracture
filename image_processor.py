import cv2
import numpy as np
from typing import Tuple, List, Optional

class ImageProcessor:
    """Handles image preprocessing operations for bone fracture detection."""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def invert_image(self, image: np.ndarray) -> np.ndarray:
        """
        Invert the image to highlight bone structures (white becomes black, black becomes white).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Inverted image
        """
        return cv2.bitwise_not(image)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image
        """
        return self.clahe.apply(image)
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if it's not already.
        
        Args:
            image: Input image (can be RGB or grayscale)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3 and image.shape[2] > 1:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.copy()
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline: convert to grayscale, enhance contrast, and invert.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (inverted_image, original_grayscale)
        """
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(gray)
        
        # Invert the image to highlight bones
        inverted = self.invert_image(enhanced)
        
        return inverted, gray
