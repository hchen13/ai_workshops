from keras.engine.saving import load_model
import cv2
import numpy as np

if __name__ == '__main__':
    hero_recognizer = load_model('hero_recognizer.h5')
    hero_recognizer._make_predict_function()

    test_image = cv2.imread('assets/test1.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(test_image, (224, 224)) / 255.0

    pred = hero_recognizer.predict(np.array([input_image]))
    print(pred.argmax())
