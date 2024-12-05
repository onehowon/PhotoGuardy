import cv2
import numpy as np
import dlib

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_face_landmarks(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray_image)

    landmarks = []
    for face in faces:
        shape = landmark_predictor(gray_image, face)
        landmarks.extend([(shape.part(i).x, shape.part(i).y) for i in range(68)])  # 전체 랜드마크
    return landmarks
