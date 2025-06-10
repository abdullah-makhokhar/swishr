"""
Basketball YOLO Model Trainer
Handles training, validation, and testing of basketball detection models
"""

import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import yaml
import torch
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of basketball-specific YOLO models"""
    
    def __init__(self, 
                 data_config_path: str,
                 base_model: str = "yolo11n.pt",
                 project_name: str = "basketball_detection",
                 experiment_name: Optional[str] = None,
                 output_dir: str = "models/basketball"):
        """
        Initialize model trainer
        
        Args:
            data_config_path: Path to data.yaml configuration file
            base_model: Base YOLO model to start training from
            project_name: Project name for tracking/logging
            experiment_name: Experiment name for this training run
            output_dir: Directory to save trained models
        """
        self.data_config_path = Path(data_config_path)
        self.base_model = base_model
        self.project_name = project_name
        self.experiment_name = experiment_name or f"basketball_train_{int(time.time())}"
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = []
        self.best_metrics = {}
        
        # Verify data config exists
        if not self.data_config_path.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_config_path}")
            
    def prepare_training_config(self,
                              epochs: int = 100,
                              batch_size: int = 16,
                              imgsz: int = 640,
                              lr0: float = 0.01,
                              lrf: float = 0.01,
                              momentum: float = 0.937,
                              weight_decay: float = 0.0005,
                              warmup_epochs: int = 3,
                              warmup_momentum: float = 0.8,
                              warmup_bias_lr: float = 0.1,
                              box_gain: float = 0.05,
                              cls_gain: float = 0.5,
                              obj_gain: float = 1.0,
                              iou_threshold: float = 0.20,
                              anchor_t: float = 4.0,
                              augment: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """
        Prepare training configuration optimized for basketball detection
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            imgsz: Image size for training
            lr0: Initial learning rate
            lrf: Final learning rate factor
            momentum: SGD momentum
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs
            warmup_momentum: Warmup momentum
            warmup_bias_lr: Warmup bias learning rate
            box_gain: Box loss gain
            cls_gain: Class loss gain  
            obj_gain: Object loss gain
            iou_threshold: IoU threshold for NMS
            anchor_t: Anchor threshold
            augment: Whether to use data augmentation
            **kwargs: Additional training parameters
            
        Returns:
            Training configuration dictionary
        """
        
        # Basketball-specific augmentation settings
        basketball_augmentation = {
            'hsv_h': 0.015,      # Hue augmentation (reduce from default 0.015)
            'hsv_s': 0.7,        # Saturation augmentation
            'hsv_v': 0.4,        # Value augmentation
            'degrees': 10.0,     # Rotation degrees (reduce for sports)
            'translate': 0.1,    # Translation fraction
            'scale': 0.5,        # Scale factor
            'shear': 2.0,        # Shear degrees
            'perspective': 0.0,  # Perspective transformation (disable for sports)
            'flipud': 0.0,       # Vertical flip probability (disable - basketballs don't flip)
            'fliplr': 0.5,       # Horizontal flip probability
            'mosaic': 1.0,       # Mosaic probability
            'mixup': 0.1,        # Mixup probability (low for object detection)
            'copy_paste': 0.1    # Copy-paste probability
        }
        
        config = {
            # Basic training parameters
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Learning rate schedule
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            
            # Loss function weights
            'box': box_gain,
            'cls': cls_gain,
            'obj': obj_gain,
            
            # NMS parameters
            'iou': iou_threshold,
            'anchor_t': anchor_t,
            
            # Data augmentation
            'augment': augment,
            **basketball_augmentation,
            
            # Optimization
            'optimizer': 'SGD',  # or 'Adam', 'AdamW'
            'cos_lr': True,      # Cosine learning rate scheduler
            'dropout': 0.0,      # Dropout (only for training)
            
            # Validation
            'val': True,
            'save': True,
            'save_period': 10,   # Save checkpoint every N epochs
            'cache': False,      # Cache images for faster training
            'workers': 8,        # Number of worker threads
            'project': str(self.output_dir),
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            
            # Additional parameters
            **kwargs
        }
        
        return config
        
    def train_model(self,
                   training_config: Optional[Dict[str, Any]] = None,
                   resume: bool = False,
                   resume_from: Optional[str] = None,
                   use_wandb: bool = False,
                   callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Train the basketball detection model
        
        Args:
            training_config: Training configuration dict
            resume: Whether to resume training from checkpoint
            resume_from: Specific checkpoint path to resume from
            use_wandb: Whether to use Weights & Biases logging
            callbacks: Custom training callbacks
            
        Returns:
            Training results and metrics
        """
        
        # Prepare training configuration
        if training_config is None:
            training_config = self.prepare_training_config()
            
        # Initialize Weights & Biases if requested
        if use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("Weights & Biases not available. Install with: pip install wandb")
                use_wandb = False
            else:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=training_config
                )
            
        try:
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            model = YOLO(self.base_model)
            
            # Set up training data
            training_config['data'] = str(self.data_config_path)
            
            logger.info(f"Starting training with config: {training_config}")
            logger.info(f"Data config: {self.data_config_path}")
            logger.info(f"Output directory: {self.output_dir / self.experiment_name}")
            
            # Start training
            start_time = time.time()
            results = model.train(**training_config)
            training_time = time.time() - start_time
            
            # Get best model path
            best_model_path = self.output_dir / self.experiment_name / "weights" / "best.pt"
            last_model_path = self.output_dir / self.experiment_name / "weights" / "last.pt"
            
            # Extract training metrics
            training_results = {
                'training_time': training_time,
                'best_model_path': str(best_model_path),
                'last_model_path': str(last_model_path),
                'experiment_name': self.experiment_name,
                'training_config': training_config
            }
            
            # Extract metrics from results if available
            if hasattr(results, 'results_dict'):
                training_results.update(results.results_dict)
            elif isinstance(results, dict):
                training_results.update(results)
                
            # Store training history
            self.training_history.append(training_results)
            
            # Save training report
            self._save_training_report(training_results)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best model saved to: {best_model_path}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if use_wandb:
                wandb.finish()
                
    def validate_model(self, 
                      model_path: str,
                      data_config: Optional[str] = None,
                      imgsz: int = 640,
                      batch_size: int = 32,
                      conf_threshold: float = 0.001,
                      iou_threshold: float = 0.6,
                      save_json: bool = True,
                      save_hybrid: bool = False) -> Dict[str, Any]:
        """
        Validate trained model on validation set
        
        Args:
            model_path: Path to trained model
            data_config: Data configuration path (uses training config if None)
            imgsz: Image size for validation
            batch_size: Validation batch size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            save_json: Save results in COCO JSON format
            save_hybrid: Save hybrid labels (labels + additional IoU info)
            
        Returns:
            Validation results and metrics
        """
        
        data_config = data_config or str(self.data_config_path)
        
        try:
            logger.info(f"Validating model: {model_path}")
            
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            validation_results = model.val(
                data=data_config,
                imgsz=imgsz,
                batch=batch_size,
                conf=conf_threshold,
                iou=iou_threshold,
                save_json=save_json,
                save_hybrid=save_hybrid,
                verbose=True
            )
            
            # Extract key metrics
            metrics = {
                'mAP50': getattr(validation_results, 'map50', 0),
                'mAP50-95': getattr(validation_results, 'map', 0),
                'precision': getattr(validation_results, 'mp', 0),
                'recall': getattr(validation_results, 'mr', 0),
                'f1_score': getattr(validation_results, 'f1', 0),
                'model_path': model_path,
                'validation_config': {
                    'imgsz': imgsz,
                    'batch_size': batch_size,
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold
                }
            }
            
            # Add class-specific metrics if available
            if hasattr(validation_results, 'maps'):
                class_names = ['basketball', 'hoop', 'person']
                for i, class_name in enumerate(class_names):
                    if i < len(validation_results.maps):
                        metrics[f'mAP50_{class_name}'] = validation_results.maps[i]
                        
            logger.info(f"Validation completed - mAP50: {metrics['mAP50']:.3f}, mAP50-95: {metrics['mAP50-95']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
            
    def test_model(self,
                  model_path: str,
                  test_images_dir: str,
                  output_dir: Optional[str] = None,
                  conf_threshold: float = 0.25,
                  iou_threshold: float = 0.45,
                  save_results: bool = True) -> Dict[str, Any]:
        """
        Test model on custom test set
        
        Args:
            model_path: Path to trained model
            test_images_dir: Directory containing test images
            output_dir: Output directory for test results
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save detection results
            
        Returns:
            Test results and performance metrics
        """
        
        output_dir = output_dir or str(self.output_dir / "test_results" / self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            logger.info(f"Testing model: {model_path}")
            logger.info(f"Test images: {test_images_dir}")
            
            # Load model
            model = YOLO(model_path)
            
            # Get test images
            test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
            
            if not test_images:
                raise ValueError(f"No test images found in {test_images_dir}")
                
            logger.info(f"Found {len(test_images)} test images")
            
            # Run inference on test images
            results = model(
                source=test_images_dir,
                conf=conf_threshold,
                iou=iou_threshold,
                save=save_results,
                save_txt=save_results,
                save_conf=save_results,
                project=output_dir,
                name="predictions",
                exist_ok=True
            )
            
            # Analyze results
            total_detections = 0
            class_detections = {0: 0, 1: 0, 2: 0}  # basketball, hoop, person
            confidence_scores = []
            
            for result in results:
                if result.boxes is not None:
                    total_detections += len(result.boxes)
                    confidence_scores.extend(result.boxes.conf.cpu().numpy().tolist())
                    
                    for cls in result.boxes.cls.cpu().numpy():
                        cls = int(cls)
                        if cls in class_detections:
                            class_detections[cls] += 1
                            
            test_metrics = {
                'total_test_images': len(test_images),
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / len(test_images),
                'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'class_detections': {
                    'basketball': class_detections[0],
                    'hoop': class_detections[1],
                    'person': class_detections[2]
                },
                'test_config': {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'model_path': model_path,
                    'test_images_dir': test_images_dir,
                    'output_dir': output_dir
                }
            }
            
            # Save test report
            self._save_test_report(test_metrics)
            
            logger.info(f"Testing completed - {total_detections} detections on {len(test_images)} images")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise
            
    def _save_training_report(self, training_results: Dict[str, Any]):
        """Save training report to file"""
        report_path = self.output_dir / f"{self.experiment_name}_training_report.yaml"
        
        with open(report_path, 'w') as f:
            yaml.dump(training_results, f, default_flow_style=False)
            
        logger.info(f"Training report saved to: {report_path}")
        
    def _save_test_report(self, test_metrics: Dict[str, Any]):
        """Save test report to file"""
        report_path = self.output_dir / f"{self.experiment_name}_test_report.yaml"
        
        with open(report_path, 'w') as f:
            yaml.dump(test_metrics, f, default_flow_style=False)
            
        logger.info(f"Test report saved to: {report_path}")
        
    def compare_models(self, model_paths: List[str], validation_data: str) -> Dict[str, Any]:
        """
        Compare multiple trained models
        
        Args:
            model_paths: List of model paths to compare
            validation_data: Validation data config path
            
        Returns:
            Comparison results
        """
        
        comparison_results = {}
        
        for model_path in model_paths:
            try:
                model_name = Path(model_path).stem
                logger.info(f"Evaluating model: {model_name}")
                
                # Validate model
                metrics = self.validate_model(model_path, validation_data)
                comparison_results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
                comparison_results[Path(model_path).stem] = {'error': str(e)}
                
        # Find best model
        best_model = None
        best_map50 = 0
        
        for model_name, metrics in comparison_results.items():
            if 'mAP50' in metrics and metrics['mAP50'] > best_map50:
                best_map50 = metrics['mAP50']
                best_model = model_name
                
        comparison_results['best_model'] = best_model
        comparison_results['best_mAP50'] = best_map50
        
        # Save comparison report
        comparison_path = self.output_dir / f"model_comparison_{int(time.time())}.yaml"
        with open(comparison_path, 'w') as f:
            yaml.dump(comparison_results, f, default_flow_style=False)
            
        logger.info(f"Model comparison saved to: {comparison_path}")
        logger.info(f"Best model: {best_model} (mAP50: {best_map50:.3f})")
        
        return comparison_results
        
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
        
    def clean_checkpoints(self, keep_best: bool = True, keep_last: bool = True):
        """
        Clean up training checkpoints to save space
        
        Args:
            keep_best: Whether to keep best.pt
            keep_last: Whether to keep last.pt
        """
        
        weights_dir = self.output_dir / self.experiment_name / "weights"
        
        if not weights_dir.exists():
            return
            
        checkpoint_files = list(weights_dir.glob("epoch*.pt"))
        
        for checkpoint in checkpoint_files:
            if checkpoint.name not in ['best.pt', 'last.pt']:
                checkpoint.unlink()
                logger.info(f"Removed checkpoint: {checkpoint}")
            elif checkpoint.name == 'best.pt' and not keep_best:
                checkpoint.unlink()
                logger.info(f"Removed best checkpoint: {checkpoint}")
            elif checkpoint.name == 'last.pt' and not keep_last:
                checkpoint.unlink()
                logger.info(f"Removed last checkpoint: {checkpoint}") 