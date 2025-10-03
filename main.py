import gradio as gr
import numpy as np
import cv2
from PIL import Image
from image_processor import ImageProcessor
from bone_isolator import BoneIsolator
from fracture_detector import FractureDetector
from visualization import FractureVisualizer

class BoneFractureDetectionApp:
    """Main application class for bone fracture detection."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.bone_isolator = BoneIsolator()
        self.fracture_detector = FractureDetector(angle_threshold=25.0, distance_threshold=8.0)
        self.visualizer = FractureVisualizer()
    
    def process_xray_image(self, input_image):
        """
        Process X-ray image and detect fractures.
        
        Args:
            input_image: Input image from Gradio
            
        Returns:
            Tuple of processed images and results
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(input_image, Image.Image):
                img_array = np.array(input_image)
            else:
                img_array = input_image.copy()
            
            # Step 1: Preprocess image (convert to grayscale, enhance contrast, invert)
            inverted_image, original_gray = self.image_processor.preprocess_image(img_array)
            
            # Step 2: Isolate bone structures
            isolated_bone, bone_mask = self.bone_isolator.isolate_bone(inverted_image)
            
            # Step 3: Detect fractures
            fracture_regions = self.fracture_detector.detect_fractures(bone_mask)
            
            # Step 4: Visualize results
            # Draw fracture boxes on the isolated bone image
            result_with_boxes = self.visualizer.draw_fracture_boxes(isolated_bone, fracture_regions)
            
            # Add summary to the result
            final_result = self.visualizer.create_result_summary(result_with_boxes, len(fracture_regions))
            
            # Create step-by-step visualization
            steps_visualization = self.visualizer.create_processing_steps_visualization(
                original_gray, inverted_image, isolated_bone, final_result
            )
            
            # Prepare summary text
            fracture_count = len(fracture_regions)
            if fracture_count == 0:
                summary = "✅ No fractures detected"
                confidence = "Analysis complete - no anomalies found"
            else:
                summary = f"⚠️ {fracture_count} potential fracture(s) detected"
                confidence = f"Found {fracture_count} suspicious region(s) requiring medical review"
            
            return (
                final_result,           # Main result with bounding boxes
                steps_visualization,    # Step-by-step process visualization
                summary,               # Summary text
                confidence            # Confidence/details text
            )
            
        except Exception as e:
            # Handle errors gracefully
            error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            error_message = f"Error processing image: {str(e)}"
            
            cv2.putText(error_img, "Error Processing Image", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(error_img, str(e)[:50], (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            return error_img, error_img, "Error", error_message

def create_gradio_interface():
    """Create and launch the Gradio interface."""
    
    app = BoneFractureDetectionApp()
    
    # Create the Gradio interface
    interface = gr.Interface(
        fn=app.process_xray_image,
        inputs=[
            gr.Image(type="numpy", label="Upload X-ray Image")
        ],
        outputs=[
            gr.Image(type="numpy", label="Fracture Detection Result"),
            gr.Image(type="numpy", label="Processing Steps Visualization"),
            gr.Text(label="Detection Summary"),
            gr.Text(label="Analysis Details")
        ],
        title="🦴 Bone Fracture Detection System",
        description="""
        **Computer Vision-Based Fracture Detection**
        
        Upload an X-ray image to detect potential bone fractures using advanced computer vision techniques:
        
        1. **Image Inversion**: Enhances bone visibility by inverting pixel values
        2. **Bone Isolation**: Removes background noise and isolates bone structures  
        3. **Fracture Detection**: Uses two methods:
           - **Greatest Diversion Theory**: Detects abnormal angles in bone structure
           - **Distance Separation**: Identifies gaps or separations in bone continuity
        4. **Visualization**: Highlights suspected fracture areas with green bounding boxes
        """,
        article="""
        ### How It Works
        
        **Processing Pipeline:**
        1. **Preprocessing**: Convert to grayscale → Enhance contrast → Invert image
        2. **Bone Isolation**: Create bone mask → Remove background → Extract main bone structure
        3. **Fracture Analysis**: 
           - Analyze bone contour for angle deviations > 25°
           - Detect distance separations > 8 pixels
        4. **Visualization**: Draw green bounding boxes around suspected fracture areas
        
        **Detection Methods:**
        - **Angle Analysis**: Measures deviations from normal bone curvature
        - **Gap Detection**: Identifies physical separations in bone structure
        
        ### Important Notes
        - This tool is for **educational and research purposes only**
        - **Not intended for medical diagnosis** - always consult healthcare professionals
        - Works best with clear, high-contrast X-ray images
        - Green boxes indicate areas requiring further medical evaluation
        
        ### Supported Image Types
        - X-ray images (DICOM, PNG, JPEG)
        - Grayscale or color images
        - Various bone types (arm, leg, hand, etc.)
        """,
        examples=[
            # You can add example images here if you have them in the project
        ],
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
