import cv2
import numpy as np
import math
from typing import List, Tuple, Optional

class FractureDetector:
    """Detects bone fractures using computer vision techniques."""
    
    def __init__(self, angle_threshold: float = 45.0, distance_threshold: float = 15.0, 
                 max_fractures: int = 3, confidence_threshold: float = 0.7):
        """
        Initialize fracture detector with stricter thresholds.
        
        Args:
            angle_threshold: Minimum angle deviation to consider as fracture (degrees) - increased from 30 to 45
            distance_threshold: Minimum distance separation to consider as fracture (pixels) - increased from 10 to 15
            max_fractures: Maximum number of fractures to return (only the most significant ones)
            confidence_threshold: Minimum confidence score to consider a detection valid
        """
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold
        self.max_fractures = max_fractures
        self.confidence_threshold = confidence_threshold
    
    def extract_bone_skeleton(self, bone_mask: np.ndarray) -> np.ndarray:
        """
        Extract the skeleton (centerline) of the bone for analysis.
        
        Args:
            bone_mask: Binary mask of the bone
            
        Returns:
            Skeletonized bone image
        """
        # Apply morphological thinning to get skeleton
        skeleton = cv2.ximgproc.thinning(bone_mask)
        return skeleton
    
    def find_bone_contour_points(self, bone_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contour points of the bone for analysis.
        
        Args:
            bone_mask: Binary mask of the bone
            
        Returns:
            List of contour points
        """
        contours, _ = cv2.findContours(bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            # Return the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.reshape(-1, 2)
        return []
    
    def calculate_angle_deviation(self, points: List[np.ndarray], window_size: int = 20) -> List[Tuple[int, float, float]]:
        """
        Calculate angle deviations along the bone contour using the greatest diversion theory.
        
        Args:
            points: List of contour points
            window_size: Size of the window for angle calculation
            
        Returns:
            List of (point_index, angle_deviation, confidence_score) tuples
        """
        if len(points) < window_size * 2:
            return []
        
        angle_deviations = []
        
        for i in range(window_size, len(points) - window_size):
            # Get three points: before, current, after
            p1 = points[i - window_size]
            p2 = points[i]
            p3 = points[i + window_size]
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = math.degrees(math.acos(cos_angle))
            
            # Calculate deviation from straight line (180 degrees)
            deviation = abs(180 - angle)
            
            confidence = min(1.0, deviation / (self.angle_threshold * 1.5))
            
            angle_deviations.append((i, deviation, confidence))
        
        return angle_deviations
    
    def find_distance_separations(self, skeleton: np.ndarray) -> List[Tuple[Tuple[int, int], float, float]]:
        """
        Find distance separations in the bone skeleton that might indicate fractures.
        
        Args:
            skeleton: Skeletonized bone image
            
        Returns:
            List of (gap_center_point, gap_distance, confidence_score) tuples
        """
        # Find endpoints and junctions in skeleton
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilate and subtract to find gaps
        dilated = cv2.dilate(skeleton, kernel, iterations=1)
        gaps = cv2.subtract(dilated, skeleton)
        
        # Find contours of gaps
        contours, _ = cv2.findContours(gaps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        separations = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:  # Filter small noise
                # Calculate the center of the gap
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate gap distance (approximate)
                    rect = cv2.boundingRect(contour)
                    gap_distance = max(rect[2], rect[3])  # Max of width and height
                    
                    confidence = min(1.0, gap_distance / (self.distance_threshold * 1.5))
                    
                    separations.append(((cx, cy), gap_distance, confidence))
        
        return separations
    
    def detect_fractures(self, bone_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect only the most significant fractures using both angle deviation and distance separation methods.
        
        Args:
            bone_mask: Binary mask of the bone
            
        Returns:
            List of bounding boxes (x, y, width, height) around the most significant suspected fracture areas
        """
        fracture_candidates = []
        
        # Method 1: Greatest diversion theory (angle analysis)
        contour_points = self.find_bone_contour_points(bone_mask)
        if len(contour_points) > 0:
            angle_deviations = self.calculate_angle_deviation(contour_points)
            
            for point_idx, deviation, confidence in angle_deviations:
                # Only consider fractures that exceed threshold AND have sufficient confidence
                if deviation > self.angle_threshold and confidence > self.confidence_threshold:
                    point = contour_points[point_idx]
                    x, y = point[0], point[1]
                    
                    # Create a bounding box around the fracture point
                    box_size = 60  # Slightly larger for better visibility
                    bbox = (
                        max(0, x - box_size // 2),
                        max(0, y - box_size // 2),
                        box_size,
                        box_size
                    )
                    
                    # Store with combined score for ranking
                    combined_score = deviation * confidence
                    fracture_candidates.append((bbox, combined_score, 'angle'))
        
        # Method 2: Distance separation analysis
        try:
            skeleton = self.extract_bone_skeleton(bone_mask)
            separations = self.find_distance_separations(skeleton)
            
            for (cx, cy), gap_distance, confidence in separations:
                # Only consider separations that exceed threshold AND have sufficient confidence
                if gap_distance > self.distance_threshold and confidence > self.confidence_threshold:
                    # Create bounding box around the separation
                    box_size = max(60, int(gap_distance * 2))  # Scale box with gap size, minimum 60
                    bbox = (
                        max(0, cx - box_size // 2),
                        max(0, cy - box_size // 2),
                        box_size,
                        box_size
                    )
                    
                    # Store with combined score for ranking
                    combined_score = gap_distance * confidence
                    fracture_candidates.append((bbox, combined_score, 'distance'))
        except Exception as e:
            print(f"Warning: Skeleton analysis failed: {e}")
        
        # Sort candidates by combined score (highest first)
        fracture_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take only the top candidates
        top_candidates = fracture_candidates[:self.max_fractures]
        
        # Extract just the bounding boxes
        fracture_regions = [candidate[0] for candidate in top_candidates]
        
        # Remove overlapping boxes from the top candidates
        fracture_regions = self._remove_overlapping_boxes(fracture_regions)
        
        # Print detection summary for debugging
        if fracture_regions:
            print(f"[v0] Detected {len(fracture_regions)} significant fractures from {len(fracture_candidates)} candidates")
            for i, (bbox, score, method) in enumerate(top_candidates[:len(fracture_regions)]):
                print(f"[v0] Fracture {i+1}: {method} method, score: {score:.2f}")
        else:
            print(f"[v0] No significant fractures detected (found {len(fracture_candidates)} candidates below threshold)")
        
        return fracture_regions
    
    def _remove_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                                 overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Remove overlapping bounding boxes.
        
        Args:
            boxes: List of bounding boxes
            overlap_threshold: Minimum overlap ratio to consider boxes as overlapping
            
        Returns:
            Filtered list of bounding boxes
        """
        if len(boxes) <= 1:
            return boxes
        
        # Calculate areas
        areas = [(box[2] * box[3]) for box in boxes]
        
        # Sort by area (largest first)
        sorted_indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        
        keep = []
        for i in sorted_indices:
            box1 = boxes[i]
            should_keep = True
            
            for j in keep:
                box2 = boxes[j]
                
                # Calculate intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[0] + box1[2], box2[0] + box2[2])
                y2 = min(box1[1] + box1[3], box2[1] + box2[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = areas[i] + areas[j] - intersection
                    
                    if intersection / union > overlap_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(i)
        
        return [boxes[i] for i in keep]
