# Usage for Albumentations library

## Menu:

<a href="#label1">1. Specific Transforms</a>

<a href="#label2">2. Integration with YOLO</a>

<a href="#label3">3. Program Template</a>



## <span id="label1">Specific transforms</span>

To effectively argument our datasets, we normally follow the 7-step approach:

1. <a href="#step1">Start with Cropping - Size normalization first.</a>
2. <a href="#step2">Geometric Transformations - HorizontalFlip,SquareSymmertry etc.</a>
3. <a href="#step3">Dropout - CoarseDropout, RandomErasing.</a>
4. <a href="#step4">Reduce Color Dependence - ToGray, ChannelDropOut.</a>
5. <a href="#step5">Affine Transformation - Affine for scale or rotation.</a>
6. <a href="#step6">Specialized transforms for specific use case.</a>
7. <a href="#step7">Normalization - Standard or sample-specific.</a>

> Key principle:
>
> 1. Add one argumentation at a time.
> 2. Fine tune parameters for each model.
> 3. Visualize argumentations.

### NOTE!

**For detection models like YOLO, we need to add bbox_params**

[albumentations.core.bbox_utils | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/core/bbox_utils/#BboxParams)

### <span id="step1">Step 1. Cropping(if Applicable)</span>

`A.RandomCrop(params)`[albumentations.augmentations.crops.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/crops/transforms/#RandomCrop)

`A.RandomResizedCrop(params)`[albumentations.augmentations.crops.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/crops/transforms/#RandomResizedCrop)

`A.RandomSizedBBoxSafeCrop(params)`[albumentations.augmentations.crops.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/crops/transforms/#RandomSizedBBoxSafeCrop)

Alternative Resizing Strategies:

`A.SmallestMaxSize(params)`(Minimum side is equal to max_size)[albumentations.augmentations.geometric.resize | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/resize/#SmallestMaxSize)

`A.LongestMaxSize(params)`(Longest side is equal to max_size)[albumentations.augmentations.geometric.resize | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/resize/#LongestMaxSize)

### <span id="step2">Step 2. Baic Geometric Transformations</span>

`A.HorizontalFlip(params)`[albumentations.augmentations.geometric.flip | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/flip/#HorizontalFlip)

`A.VerticalFlip(params)`[albumentations.augmentations.geometric.flip | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/flip/#VerticalFlip)

`A.SquareSymmetry(params)`[albumentations.augmentations.geometric.flip | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/flip/#SquareSymmetry)

`A.Rotate(params)`[albumentations.augmentations.geometric.rotate | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/rotate/#Rotate)

### <span id="step3">Step 3.Add Dropout</span>

`A.CoarseDropout(params)`[albumentations.augmentations.dropout.coarse_dropout | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/dropout/coarse_dropout/#CoarseDropout)

`A.Erasing(params)`[albumentations.augmentations.dropout.coarse_dropout | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/dropout/coarse_dropout/#Erasing)

`A.ConstrainedCoarseDropout(params)`[albumentations.augmentations.dropout.coarse_dropout | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/dropout/coarse_dropout/#ConstrainedCoarseDropout)

> Visualize the results to ensure you aren't removing too much information.

### <span id="step4">Step 4. Reduce Reliance on Color</span>

`A.ChannelShuffle(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#ChannelShuffle)

`A.CLAHE(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#CLAHE)

`A.ColorJitter(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#ColorJitter)

`A.HueSaturationValue(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#HueSaturationValue)

`A.ToGray(params)`...([albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/))

### <span id="step5">Step 5. Affine Transformations</span>

`A.Affine(params)`[albumentations.augmentations.geometric.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/transforms/#Affine)

Additionally, `A.Perspective(params)` is another powerful geometric transform.

[albumentations.augmentations.geometric.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/transforms/#Perspective)

### <span id="step6">Step 6. Specialized transformations</span>

`A.Mosaic(params)`[albumentations.augmentations.mixing.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/mixing/transforms/#Mosaic)

`A.GaussNoise(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#GaussNoise)

`A.RandomBrightnessContrast(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#RandomBrightnessContrast)

`A.RandomSunFlare(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#RandomSunFlare)

`A.MotionBlur(params)`[albumentations.augmentations.blur.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/blur/transforms/#MotionBlur)

...([albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/))

### <span id="step7">Step 7. Normalization</span>

`A.Normalize(params)`[albumentations.augmentations.pixel.transforms | Albumentations](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/#Normalize)



## <span id="label2">Intergration with YOLO</span>

Write training programs that fits YOLO.

[Documentation](https://albumentations.ai/docs/examples/example-ultralytics/)



## <span id="label3">Program Template:</span>

```python
# import necessary libraries
import cv2
import albumentations as A

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # convert picture from BGR to RGB
    image = cv2.cvtColor(frame, cv2.Color_BGR2RGB)
    transform = A.HorizontalFlip(p = 0.5)# any supported transformation
    argumented = transform(image = image)['image']
    # transform the image back to BGR
    argumented = cv2.cvtColor(argumented, cv2.Color_RGB2BGR)
    cv2.imshow('Argumented', argumented)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
```

- Note the three 'image' in `argumented = transtorm(image = image)['image']`
  1. The actual name of the parameter.
  2. The image to transform.
  3. The expression return a dictionary which includes the key named 'image'

