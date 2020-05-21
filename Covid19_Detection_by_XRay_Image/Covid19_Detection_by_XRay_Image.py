import cv2
import tensorflow.python.keras as tfk

def prepare(filepath):
    IMG_SIZE = 70  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  
value = input("Please select an image: ")
try:
    model = tfk.models.load_model("Covid-19_DataSet.model")
    try:
        prediction = model.predict([prepare(value)])
        if prediction>0.8:
            print("Covid_19_Negative")
        else: 
            print("Covid_19_Positive")
    except Exception as E:
        print("No file found")
except Exception as E:
        print("No model found")
