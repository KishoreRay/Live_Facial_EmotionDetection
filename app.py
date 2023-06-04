import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained model
model = load_model('facial_emotion_model.h5')

def predict_emotion(face_image):
    # Resize the input image to match the input size of the model
    resized_image = cv2.resize(face_image, (48, 48))
    # Convert the image to grayscale and normalize its pixel values
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray_image = np.reshape(gray_image, (48, 48, 1))
    # Stack the grayscale image three times along the last axis to create a three-channel image
    rgb_image = np.concatenate([gray_image]*3, axis=-1)
    rgb_image = np.reshape(rgb_image, (1, 48, 48, 3)) / 255.0
    # Use the trained model to predict the emotion in the image
    emotion_probabilities = model.predict(rgb_image)[0]
    # Get the index of the predicted emotion with the highest probability
    predicted_emotion_index = np.argmax(emotion_probabilities)
    # Define a dictionary to map the emotion index to its corresponding label
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    # Return the predicted emotion label and its corresponding probability
    return emotion_labels[predicted_emotion_index], emotion_probabilities[predicted_emotion_index]

# Open the default webcam and start capturing frames
cap = cv2.VideoCapture(0)
# Define the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        i =self.i+1
        
        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Extract the face image from the frame
            face_image = img[y:y+h, x:x+w]
            # Use the trained model to predict the emotion in the face image
            predicted_emotion, emotion_probability = predict_emotion(face_image)
            # Display the predicted emotion label and its probability on the frame
            cv2.putText(img, predicted_emotion + ' ' + str(round(emotion_probability, 2)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Display the frame with the predicted emotion labels
        return img
    
st.header("Webcam Live Feed")
st.write("Click on start to use webcam and detect your face emotion")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    
    
    
   # st.title("Webcam Live Feed")
    #st.write("Click the button below to start the webcam.")

    # Create a VideoCapture object
   # cap = cv2.VideoCapture(0)
    #FRAME_WINDOW = st.image([])


    # Read until video is stopped
    #while True:
        # _, frame = cap.read()
         #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         #FRAME_WINDOW.image('Facial Emotion Recognition',frame)
    
         #if cv2.waitKey(1) & 0xFF == ord('q'):
           # break

    # Release the webcam and close all windows
   # cap.release()
    #cv2.destroyAllWindows()

#if __name__ == "__main__":
 #   main()
