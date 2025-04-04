from ultralytics import YOLO

def train_model():
    # Path to the YOLOv8 segmentation model weights
    model_path = "yolov8n-seg.pt"

    # Path to the dataset configuration file
    data_yaml_path = "dataset/data.yaml"

    # Initialize the YOLO model
    model = YOLO(model_path)

    # Define data augmentation parameters
    augmentation_params = {
        "hsv_h": 0.015,  # Hue shift
        "hsv_s": 0.7,    # Saturation shift
        "hsv_v": 0.4,    # Value (brightness) shift
        "degrees": 5.0,  # Rotation
        "translate": 0.1,  # Translation
        "scale": 0.5,    # Scaling
        "shear": 0.5,    # Shearing
        "perspective": 0.0,  # Perspective distortion
        "flipud": 0.0,   # Vertical flip
        "fliplr": 0.5,   # Horizontal flip
        "mosaic": 1.0,   # Mosaic augmentation
        "mixup": 0.0     # Mixup augmentation
    }

    # Train the model with augmentation
    model.train(
        data=data_yaml_path,
        epochs=200,
        imgsz=640,
        lr0=0.01,                # Initial learning rate
        momentum=0.937,          # Momentum
        weight_decay=0.0005,     # Weight decay
        batch=32,                 # Batch size
        workers=10,                # Number of data loader workers
        project="medical_segmentation",  # Project folder for saving results
        name="exp_augmented",     # Experiment name
        device=0,                 # GPU device
        augment=True,             # Enable augmentations
        **augmentation_params     # Apply augmentation parameters
    )

if __name__ == '__main__':
    train_model()
