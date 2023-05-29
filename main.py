import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import tensorflow as tf

from tensorflow import keras
import cv2 as cv

DEBUG : bool = False

img_height : int = 64
img_width : int = 64
org = (50, 50)

loaded_model = tf.keras.models.load_model('model')
print(loaded_model.summary())

class_names : list = ['apple fruit',
                    'banana fruit',
                    'cherry fruit',
                    'chickoo fruit',
                    'grapes fruit',
                    'kiwi fruit',
                    'mango fruit',
                    'orange fruit',
                    'strawberry fruit']

camara : cv.VideoCapture = cv.VideoCapture(0)

while True:
    if DEBUG:
        matrix = cv.imread('imagenes/banana.jpg')
    else:
        _, matrix = camara.read()
    
    matrix_to_model = matrix
    
    matrix_to_model = cv.resize(matrix_to_model, (img_width, img_height))
    matrix_to_model = matrix_to_model.reshape(1, img_width, img_height, 3)
    
    
    #matrix_to_model = tf.expand_dims(matrix_to_model, 1)
    print(f'Shape de la imagen: {matrix_to_model.shape}')
    predictions = loaded_model.predict(matrix_to_model)
    
    score = tf.nn.softmax(predictions[0])
    label = class_names[np.argmax(predictions)]
    print(f'Predicciones: {score}')
    print(f'Score: {np.argmax(score)}')
    print(f'Etiqueta: {label}')
    
    cv.putText(matrix,
               label,
               org,
               cv.FONT_HERSHEY_SIMPLEX,
               2,
               (255,0,0),
               3,
               cv.LINE_AA)
    
    cv.imshow('Camara', matrix)
    #os.system('clear')
    
    if cv.waitKey(10) == ord('x'):
        break
    
camara.release()
cv.destroyAllWindows()