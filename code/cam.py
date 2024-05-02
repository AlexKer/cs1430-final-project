import cv2

def camcapture():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces and draw rectangale around face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Placeholder text for  label
            # reference: https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            label = "Happy"
            # label_position = (x, y - 10)
            # # Put placeholder text above the rectangle
            # cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Set the text color and background color
            text_color = (255, 255, 255)
            background_color = (255, 0, 0)
            font_scale = 0.8
            thickness = 2
            
            # Calculate text width & height to draw the background rectangle
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            width, height = text_size[0]
            
            # Draw background rectangle for text
            img = cv2.rectangle(img, (x, y - 30), (x + width, y), background_color, -1)
            # Put text above the rectangle
            img = cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)



        cv2.imshow('Face Detection', img)

        #  esc key to stop
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

camcapture()
