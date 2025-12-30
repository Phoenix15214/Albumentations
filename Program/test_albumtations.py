import albumentations as A
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Original', frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Multiple transformation
    transform = A.Compose([A.HorizontalFlip(p = 0.5), A.RandomBrightnessContrast(p = 0.2), A.Rotate(limit = 30, p = 0.5)])
    argumented = transform(image=image)['image']
    argumented = cv2.cvtColor(argumented, cv2.COLOR_RGB2BGR)
    cv2.imshow('Argumented', argumented)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()