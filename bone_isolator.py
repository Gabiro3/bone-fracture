import cv2
import numpy as np
from typing import Tuple, List

class BoneIsolator:
    """Isolates bone structures from X-ray images by removing background."""
    
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)
    
    def create_bone_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask to isolate bone structures from the background.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary mask where bones are white (255) and background is black (0)
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Use Otsu's thresholding to separate bone from background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the mask
        # Opening to remove small noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
        
        # Closing to fill small holes in bones
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        
        return closing
    
    def isolate_largest_bone(self, mask: np.ndarray) -> np.ndarray:
        """
        Isolate the largest bone structure from the mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask containing only the largest bone structure
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Find the largest contour (main bone)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest bone
        isolated_mask = np.zeros_like(mask)
        cv2.drawContours(isolated_mask, [largest_contour], 0, 255, -1)
        
        return isolated_mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply the bone mask to create a binary image with white bones on black background.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Binary image where bones are white (255) and background is black (0)
        """
        # Return the mask itself since it already has bones as white (255) and background as black (0)
        return mask

    def isolate_bone(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete bone isolation pipeline.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (binary_bone_image, bone_mask) - both are the same binary image
        """
        # Create bone mask
        mask = self.create_bone_mask(image)
        
        # Isolate the largest bone structure
        isolated_mask = self.isolate_largest_bone(mask)
        
        binary_bone_image = isolated_mask.copy()
        
        return binary_bone_image, isolated_mask
