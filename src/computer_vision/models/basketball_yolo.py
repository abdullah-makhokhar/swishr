"""
Custom Basketball YOLO Model
Basketball-specific YOLO implementation with integrated tracking
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

class BasketballYOLO:
    """Basketball-specific YOLO model with integrated tracking"""
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 model_size: str = "n",
                 device: str = "auto",
                 track_persist: bool = True,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.7):
        """
        Initialize basketball YOLO model
        
        Args:
            model_path: Path to custom trained model. If None, uses base YOLO11
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x') 
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            track_persist: Whether to persist tracking across frames
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.track_persist = track_persist
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Basketball-specific class mappings
        self.basketball_classes = {
            0: 'basketball',
            1: 'hoop', 
            2: 'person'
        }
        
        # Performance tracking
        self.detection_history = []
        self.tracking_history = {}
        
        # Initialize model
        self._load_model()
        
    def _load_model(self):
        """Load and configure YOLO model"""
        try:
            if self.model_path and Path(self.model_path).exists():
                # Load custom trained model
                logger.info(f"Loading custom basketball model: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # Load base YOLO11 model and prepare for transfer learning
                logger.info(f"Loading base YOLO11{self.model_size} model")
                self.model = YOLO(f"yolo11{self.model_size}.pt")
                
            # Configure model settings
            self.model.fuse()  # Fuse layers for faster inference
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            logger.info(f"Basketball YOLO model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def detect(self, 
               image: np.ndarray, 
               conf: Optional[float] = None,
               iou: Optional[float] = None,
               classes: Optional[List[int]] = None,
               verbose: bool = False) -> Results:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array
            conf: Confidence threshold override
            iou: IoU threshold override  
            classes: Specific classes to detect
            verbose: Whether to print verbose output
            
        Returns:
            YOLO Results object
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        try:
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                classes=classes,
                verbose=verbose,
                device=self.device
            )
            
            # Store detection for analysis
            if results and len(results) > 0:
                self.detection_history.append({
                    'timestamp': cv2.getTickCount(),
                    'detections': len(results[0].boxes) if results[0].boxes is not None else 0,
                    'conf_scores': results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes is not None else [],
                    'classes': results[0].boxes.cls.cpu().numpy().tolist() if results[0].boxes is not None else []
                })
                
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
            
    def track(self, 
              image: np.ndarray,
              conf: Optional[float] = None,
              iou: Optional[float] = None,
              classes: Optional[List[int]] = None,
              persist: Optional[bool] = None,
              tracker: str = "bytetrack.yaml",
              verbose: bool = False) -> Results:
        """
        Track objects in image with YOLO built-in tracking
        
        Args:
            image: Input image as numpy array
            conf: Confidence threshold override
            iou: IoU threshold override
            classes: Specific classes to detect  
            persist: Whether to persist tracking
            tracker: Tracker configuration file
            verbose: Whether to print verbose output
            
        Returns:
            YOLO Results object with tracking IDs
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        persist = persist if persist is not None else self.track_persist
        
        try:
            results = self.model.track(
                image,
                conf=conf,
                iou=iou,
                classes=classes,
                persist=persist,
                tracker=tracker,
                verbose=verbose,
                device=self.device
            )
            
            # Update tracking history
            if results and len(results) > 0 and results[0].boxes is not None:
                frame_tracks = {}
                boxes = results[0].boxes
                
                if hasattr(boxes, 'id') and boxes.id is not None:
                    for i, track_id in enumerate(boxes.id.cpu().numpy()):
                        track_id = int(track_id)
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        frame_tracks[track_id] = {
                            'bbox': bbox.tolist(),
                            'confidence': conf,
                            'class': cls,
                            'class_name': self.basketball_classes.get(cls, f'unknown_{cls}'),
                            'timestamp': cv2.getTickCount()
                        }
                        
                        # Update tracking history
                        if track_id not in self.tracking_history:
                            self.tracking_history[track_id] = []
                        self.tracking_history[track_id].append(frame_tracks[track_id])
                        
                        # Keep only recent history (last 30 frames)
                        if len(self.tracking_history[track_id]) > 30:
                            self.tracking_history[track_id] = self.tracking_history[track_id][-30:]
                            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return None
            
    def detect_basketball_specific(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Basketball-specific detection with enhanced filtering
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with basketball-specific detections
        """
        # Run standard tracking
        results = self.track(image, classes=[0, 1, 2])  # basketball, hoop, person
        
        if not results or results.boxes is None:
            return {
                'basketballs': [],
                'hoops': [],
                'persons': [],
                'frame_info': {
                    'total_detections': 0,
                    'basketball_count': 0,
                    'hoop_count': 0,
                    'person_count': 0
                }
            }
            
        # Parse results by class
        basketballs = []
        hoops = []
        persons = []
        
        boxes = results.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            track_id = int(boxes.id[i].cpu().numpy()) if hasattr(boxes, 'id') and boxes.id is not None else None
            
            detection = {
                'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                'confidence': conf,
                'track_id': track_id,
                'class_id': cls,
                'class_name': self.basketball_classes.get(cls, f'unknown_{cls}')
            }
            
            # Enhanced basketball filtering
            if cls == 0:  # basketball
                # Additional validation for basketball detections
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                # Basketball should be roughly circular (aspect ratio close to 1)
                if 0.7 <= aspect_ratio <= 1.3 and conf > 0.3:
                    detection['aspect_ratio'] = aspect_ratio
                    detection['size'] = (width, height)
                    basketballs.append(detection)
                    
            elif cls == 1:  # hoop
                # Hoop validation - should be wider than tall
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                if aspect_ratio > 0.5 and conf > 0.25:  # More lenient for hoops
                    detection['aspect_ratio'] = aspect_ratio
                    detection['size'] = (width, height)
                    hoops.append(detection)
                    
            elif cls == 2:  # person
                # Person validation - should be taller than wide
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                if aspect_ratio < 1.5 and conf > 0.4:  # People are typically taller
                    detection['aspect_ratio'] = aspect_ratio  
                    detection['size'] = (width, height)
                    persons.append(detection)
                    
        return {
            'basketballs': basketballs,
            'hoops': hoops, 
            'persons': persons,
            'frame_info': {
                'total_detections': len(basketballs) + len(hoops) + len(persons),
                'basketball_count': len(basketballs),
                'hoop_count': len(hoops),
                'person_count': len(persons)
            }
        }
        
    def get_basketball_trajectory(self, track_id: int, window_size: int = 10) -> Optional[List[Tuple[float, float]]]:
        """
        Get basketball trajectory for a specific track ID
        
        Args:
            track_id: Tracking ID of basketball
            window_size: Number of recent positions to return
            
        Returns:
            List of (x, y) center positions or None if track not found
        """
        if track_id not in self.tracking_history:
            return None
            
        history = self.tracking_history[track_id][-window_size:]
        trajectory = []
        
        for frame_data in history:
            if frame_data['class'] == 0:  # Only basketball
                bbox = frame_data['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                trajectory.append((center_x, center_y))
                
        return trajectory if trajectory else None
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        if not self.detection_history:
            return {}
            
        recent_detections = self.detection_history[-100:]  # Last 100 frames
        
        total_detections = sum(d['detections'] for d in recent_detections)
        avg_detections = total_detections / len(recent_detections) if recent_detections else 0
        
        all_conf_scores = []
        class_counts = {0: 0, 1: 0, 2: 0}  # basketball, hoop, person
        
        for detection in recent_detections:
            all_conf_scores.extend(detection['conf_scores'])
            for cls in detection['classes']:
                if cls in class_counts:
                    class_counts[int(cls)] += 1
                    
        avg_confidence = np.mean(all_conf_scores) if all_conf_scores else 0
        
        return {
            'avg_detections_per_frame': avg_detections,
            'avg_confidence': avg_confidence,
            'total_frames_processed': len(self.detection_history),
            'active_tracks': len(self.tracking_history),
            'class_distribution': {
                'basketball': class_counts[0],
                'hoop': class_counts[1], 
                'person': class_counts[2]
            }
        }
        
    def reset_tracking(self):
        """Reset tracking history and performance stats"""
        self.tracking_history.clear()
        self.detection_history.clear()
        logger.info("Tracking history reset")
        
    def save_model(self, save_path: str):
        """Save the trained model"""
        try:
            if hasattr(self.model, 'save'):
                self.model.save(save_path)
                logger.info(f"Model saved to {save_path}")
            else:
                logger.warning("Model does not support saving")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def export_model(self, format: str = "onnx", **kwargs):
        """Export model to different formats"""
        try:
            self.model.export(format=format, **kwargs)
            logger.info(f"Model exported to {format} format")
        except Exception as e:
            logger.error(f"Failed to export model: {e}") 