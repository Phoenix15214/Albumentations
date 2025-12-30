# Usage for Albumentations library

## Program Template:

```python
# import necessary libraries
import cv2
import albumtations as A

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



## Specific transforms

1. Horizontal Flip:

   `transform = A.HorizontalFlip(p = 0.5)`

   - Flip the image horizontally.

   - 'p' refers to the probability to transform.

2. Vertical Flip:

   `transform = A.VerticalFlip(p = 0.5)`

   - Same as Horizontal Flip.

3. Rotate:

   `transform = A.Rotate(limit = (-30, 30), p = 0.7)`

4. Random Scaling:

   `transform = A.RandomScale(scale_limit=(-0.3, 0.3), p = 0.8)`

   - The scale_limit limits the scale of the transformed image in between 70% to 130% of the original  image.

5. Random Crop(Cut):

   `transform = A.RandomCrop(height = 400, width = 400, p = 0.9)`

   - height, width refers to the cropped size.

6. Random Brightness and Contrast:

   `transform = A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1)`

7. Random Color Shift(include color, saturation and brightness):

   `transform = A.HueSaturationValue(hue_shift_limit = 30, saturation_shift_limit = 30, value_shift_limit = 20, p = 1)`

8. Multiple transformations:

   `transform = A.Compose([A.HorizontalFlip(p = 0.5), A.RandomBrightnessContrast(p = 0.2), A.Rotate(limit = 30, p = 0.5)])`
