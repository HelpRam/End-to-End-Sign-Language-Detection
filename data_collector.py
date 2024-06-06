import os
import cv2
import time
import uuid

IMAGE_PATH = "CollectedImages"

labels = ['Hello', 'Yes', 'NO', 'Thanks', 'I LOVE YOU', 'Please']
number_of_images = 5

# Create directories for each label
for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)

    print(f"Collecting images for {label}")
    time.sleep(3)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        continue

    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue

        imagename = os.path.join(IMAGE_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
