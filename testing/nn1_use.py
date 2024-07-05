import tensorflow as tf
from numpy import asarray, array, float32
from PIL import Image
import cv2
from os import listdir, remove

class_names = ['lebron', 'michael']
dir = "images_to_test"
crop_offset = 15
size_x, size_y = 64, 64

def convert_image_input():
    
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #clearing the test folder
    for img in listdir(dir):
        remove(img)
    
    # convert images to jpeg, because cv2 won't read anything else
    for y, image_name in enumerate(listdir("input")):
        img = Image.open(f"input\\{image_name}")
        img = img.convert("RGB")
        img.save(f"converted\\{y}.jpeg")
    
    #extract faces from given images in <input> directory
    for counter, image_name in enumerate(listdir("converted")):
        
        img = cv2.imread(f"converted\\{image_name}")
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )
        
        if len(faces) == 0:
            print(f"Couldn't extract faces from {image_name}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        for face in faces:
            
            #crop face from image
            img_save = img.crop((face[0]-crop_offset, face[1]-crop_offset, face[0]+face[2]+crop_offset, face[1]+face[3]+crop_offset))
            #scale it down to at least 64x64
            img_save.thumbnail((size_x, size_y))
            #crop it to exactly 64x64
            img_save = img_save.crop((0, 0, size_x, size_y))
            
            img_save.save(f"images_to_test\\{counter}.jpeg")

def prepare(filepath):

    img = asarray(Image.open(f"{filepath}")) # opens and converts file to ndarray (used in NN)
    img = img / 255.0
    return img.reshape(-1, 64, 64, 3)

def get_all_data():

    imgs, labels = [], []
    
    for x in listdir(dir):
        
        img = asarray(Image.open(f"{dir}\\{x}")) # opens and converts file to ndarray (used in NN)
        img = img / 255.0
        imgs.append(img)
    return array(imgs)

def main():
    model = tf.keras.models.load_model("nn1.model")

    img = get_all_data()
    prediction = model.predict(img)
    
    print(prediction)

if __name__ == "__main__":
    convert_image_input()
    main()
