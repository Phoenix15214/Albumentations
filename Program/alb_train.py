import torch
import albumentations as A
import cv2

from functools import partial

from ultralytics import YOLO
from ultralytics.data.augment import Albumentations as UltralyticsAlbumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer

import argparse

cv2.setNumThreads(0)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training')
    
    parser.add_argument('--data', type=str, required=True, help='.yaml file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--workers', type=int, default=4, help='Workers')
    
    
    return parser.parse_args()




class CustomAlbumentations(UltralyticsAlbumentations):
    def __init__(self, transform, contains_spatial: bool, p=1.0):
        super().__init__(p)
        # replace Ultralytics predifined transforms with custom
        self.transform = transform
        self.transform.set_random_seed(torch.initial_seed())
        self.contains_spatial = contains_spatial

    def __call__(self, labels):
        labels = super().__call__(labels)
        if "cls" in labels:
            labels["cls"] = labels["cls"].reshape(-1, 1)
        return  labels

    def __repr__(self):
        return str(self.transform)

class CustomTrainer(DetectionTrainer):
    """
    Custom trainer that replaces default Ultralytics augmentations with custom Albumentations.
    
    This trainer intercepts the dataset creation process and replaces the default
    Ultralytics augmentations with our custom Albumentations transforms.
    """
    
    def __init__(self, custom_albumentations_transforms, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.custom_albumentations_transforms = custom_albumentations_transforms
        self.replacement_logged = False  # To avoid duplicate logging

    def _close_dataloader_mosaic(self):
        """
        Called when mosaic augmentation is closed (typically at later epochs).
        We need to replace augmentations here too because the dataset gets recreated.
        """
        super()._close_dataloader_mosaic()
        self.__customize_albumentations_transforms(self.train_loader.dataset)

    def __customize_albumentations_transforms(self, dataset):
        """
        Replace default Ultralytics augmentations with our custom ones.
        
        This method iterates through the dataset's transforms and replaces any
        Ultralytics Albumentations instance with our custom one.
        """
        transforms = dataset.transforms.tolist()
        for i, t in enumerate(transforms):
            if isinstance(t, UltralyticsAlbumentations):
                # Log the replacement clearly (only once to avoid spam)
                if not self.replacement_logged:
                    print("\n" + "="*80)
                    print("🔄 REPLACING DEFAULT ULTRALYTICS AUGMENTATIONS")
                    print("="*80)
                    print(f"❌ REMOVING Default Ultralytics augmentations:")
                    print(f"   {t}")
                    print(f"\n✅ APPLYING Custom Albumentations:")
                    print(f"   {self.custom_albumentations_transforms}")
                    print("="*80 + "\n")
                    self.replacement_logged = True
                
                # Replace with custom Albumentations instance
                transforms[i] = self.custom_albumentations_transforms

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build dataset and immediately replace augmentations.
        
        This is called during training initialization, so we replace
        augmentations right when the dataset is created.
        """
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        self.__customize_albumentations_transforms(dataset)
        return dataset

colors = [0, 30, 80, 117, 180, 255]

def main():
    args = parse_args()
    # Set the target image size for training
    image_size = args.imgsz

    # Define spatial augmentations with bounding box support
    # Note the bbox_params - this is CRUCIAL for spatial transforms!
    transform = A.Compose([
        # Spatial transforms that affect bounding boxes
        A.RandomSizedBBoxSafeCrop(height=args.imgsz, width=args.imgsz, erosion_rate=0.2, p=0.5),
        A.OneOf(
            [
                A.HorizontalFlip(p=1.0),
                A.SquareSymmetry(p=1.0),
            ],
            p=0.3,
        ),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.1),
        
        A.OneOf([
            A.ConstrainedCoarseDropout(
                num_holes_range=(2,4),
                hole_height_range=(0.1,0.3),
                hole_width_range=(0.1,0.3),
                bbox_labels=[1,2,3,4,5,6,7,8],
                fill=f
            )for f in colors
        ], p=0.3),

        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.OneOf(
            [
                A.ChannelShuffle(p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ],
            p=0.3
        ),
        # A.Affine(
        #     scale=(0.8, 1.2),
        #     rotate=(-15, 15),
        #     # shear={"x": (-10, 10), "y": (-5, 5)},
        #     p=0.3
        # ),
        # A.Perspective(
        #     scale=(0.1,0.2),
        #     keep_size=True,
        #     fit_output=False,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     p=0.6
        # ),
        # A.Mosaic(
        #     target_size=(args.imgsz,args.imgsz),
        #     cell_shape=(args.imgsz,args.imgsz),
        #     p=0.3
        # ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomSunFlare(flare_roi=(0.1, 0, 0.9, 0.3),
                                 angle_range=(0.25, 0.75),
                                 num_flare_circles_range=(5, 15),
                                 src_radius=200,
                                 src_color=(255, 200, 100),
                                 method="physics_based",
                                 p=1.0),
                A.ISONoise(color_shift=(0.01,0.05), intensity=(0.3,0.5),p=1.0),
                
            ],
            p=0.3,
        ),
        A.MotionBlur(
            blur_limit=(9,15),
            angle_range=(0, 180),
            direction_range=(0.2, 0.5),
            p=0.3
        ),
        # A.Normalize(
        #     mean=(0,0,0),
        #     std=(1,1,1),
        #     max_pixel_value=255,
        #     p=1.0
        # ),
    ], bbox_params=A.BboxParams(
        format="yolo",  # YOLO format: [x_center, y_center, width, height] normalized
        min_visibility=0.3,
        label_fields=["class_labels"]  # Field name for class labels
    ))

    # Create custom augmentations wrapper WITH spatial support
    custom_albumentations_transforms = CustomAlbumentations(
        transform, 
        contains_spatial=True  # THIS IS TRUE because we have spatial transforms
    )

    print("📋 Custom SPATIAL augmentations we want to apply:")
    for aug in transform.transforms:
        print(f"   • {aug}")
    print(f"\n📦 BBox params: YOLO format with label_fields=['class_labels']")

    print("\n⚠️ WATCH THE OUTPUT BELOW:")
    print("1. First you'll see: 'albumentations: Blur(p=0.01, blur_limit=(3, 7))...' - DEFAULT message")
    print("2. Then you'll see: '🔄 REPLACING DEFAULT ULTRALYTICS AUGMENTATIONS' - our REPLACEMENT")
    print("3. This confirms our custom SPATIAL augmentations are active!\n")

    # Initialize model
    model = YOLO(f'yolo11{args.model_size}.pt')  # Using nano model for faster testing

    # Train with custom spatial augmentations
    model.train(
        data=args.data, 
        epochs=args.epochs,  # Just 2 epochs for demonstration
        imgsz=image_size,  # Use the defined image size
        batch=args.batch,  # Small batch size for testing
        trainer=partial(CustomTrainer, custom_albumentations_transforms=custom_albumentations_transforms),
        device=args.device,  # Use GPU for training
        workers=args.workers,  # Disable multiprocessing for clearer output
    )

if __name__ == "__main__":
    main()