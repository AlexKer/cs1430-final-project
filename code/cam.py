import cv2
import torch
import numpy as np
# from model import VGGModel
from torchvision.transforms import Grayscale, ToTensor

# load training model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VGGModel(learning_rate=0.001)
# model.load_state_dict(torch.load('emotion_vgg_model.pth'))
# model.to(device)
# model.eval()

emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def camcapture():
    
    # capture video from webcam.
    cap = cv2.VideoCapture(-1)

    while True:
        # Read the frame
        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces and draw rectangale around face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Extract the face region from the gray image
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (224, 224))  # Resize to the input size expected by the model
            # tensor_image = ToTensor()(face_roi).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

            # Predict the emotion
            # prediction = model(tensor_image)
            # predicted_emotion = emotion_labels[torch.argmax(prediction).item()]

            # Draw rectangle around face and add label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Text for the label
            label = "happy"
            text_color = (255, 255, 255)
            background_color = (255, 0, 0)
            font_scale = 0.8
            thickness = 2

            # text size for the background rectangle
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            width, height = text_size

            # Background rectangle with text inside
            cv2.rectangle(img, (x, y - 30), (x + width, y), background_color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # Display the resulting frame
        cv2.imshow('Face Detection', img)

        # Press 'ESC' to exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

camcapture()