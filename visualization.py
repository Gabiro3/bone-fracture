import cv2
import numpy as np
from typing import List, Tuple

class FractureVisualizer:
    """Handles visualization of fracture detection results."""
    
    def __init__(self):
        self.fracture_color = (0, 255, 0)  # Green color for bounding boxes
        self.box_thickness = 2
        self.text_color = (0, 255, 0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_thickness = 2
    
    def draw_fracture_boxes(self, image: np.ndarray, 
                           fracture_regions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw green bounding boxes around detected fracture regions.
        
        Args:
            image: Input image (RGB or grayscale)
            fracture_regions: List of bounding boxes (x, y, width, height)
            
        Returns:
            Image with fracture boxes drawn
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            result_image = image.copy()
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(fracture_regions):
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), 
                         self.fracture_color, self.box_thickness)
            
            # Add label
            label = f"Fracture {i + 1}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.text_thickness)[0]
            
            # Draw label background
            cv2.rectangle(result_image, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         self.fracture_color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x, y - 5), 
                       self.font, self.font_scale, (0, 0, 0), self.text_thickness)
        
        return result_image
    
    def create_result_summary(self, image: np.ndarray, 
                            fracture_count: int) -> np.ndarray:
        """
        Add summary text to the image showing fracture detection results.
        
        Args:
            image: Input image
            fracture_count: Number of detected fractures
            
        Returns:
            Image with summary text
        """
        result_image = image.copy()
        
        # Prepare summary text
        if fracture_count == 0:
            summary = "No fractures detected"
            text_color = (0, 255, 0)  # Green for no fractures
        else:
            summary = f"{fracture_count} potential fracture(s) detected"
            text_color = (255, 0, 0)  # Red for fractures detected
        
        # Get text size for background
        text_size = cv2.getTextSize(summary, self.font, self.font_scale, self.text_thickness)[0]
        
        # Draw background rectangle
        cv2.rectangle(result_image, 
                     (10, 10), 
                     (20 + text_size[0], 20 + text_size[1]), 
                     (255, 255, 255), -1)
        
        # Draw summary text
        cv2.putText(result_image, summary, (15, 15 + text_size[1]), 
                   self.font, self.font_scale, text_color, self.text_thickness)
        
        return result_image
    
    def create_processing_steps_visualization(self, original: np.ndarray,
                                            inverted: np.ndarray,
                                            isolated: np.ndarray,
                                            result: np.ndarray) -> np.ndarray:
        """
        Create a combined visualization showing all processing steps.
        
        Args:
            original: Original image
            inverted: Inverted image
            isolated: Isolated bone image
            result: Final result with fracture boxes
            
        Returns:
            Combined visualization image
        """
        # Ensure all images are RGB
        images = []
        titles = ["Original", "Inverted", "Isolated Bone", "Fracture Detection"]
        
        for img in [original, inverted, isolated, result]:
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img.copy()
            images.append(img_rgb)
        
        # Resize images to same size
        target_height = 300
        resized_images = []
        for img in images:
            aspect_ratio = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        
        # Add titles to images
        titled_images = []
        for img, title in zip(resized_images, titles):
            # Add title space
            title_height = 30
            titled_img = np.ones((img.shape[0] + title_height, img.shape[1], 3), dtype=np.uint8) * 255
            titled_img[title_height:, :] = img
            
            # Add title text
            text_size = cv2.getTextSize(title, self.font, self.font_scale, self.text_thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            cv2.putText(titled_img, title, (text_x, 20), 
                       self.font, self.font_scale, (0, 0, 0), self.text_thickness)
            
            titled_images.append(titled_img)
        
        # Combine images horizontally
        combined = np.hstack(titled_images)
        
        return combined
