from PIL import Image
import cv2
from os import listdir, chdir

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
dir = "lebron james"
dir = "michael jordan"
new_filename = dir.split()[0]

def get_filenames():
    fns = listdir(dir)
    chdir(dir)
    return fns
    
def get_image(filename):
    return cv2.imread(filename)

def get_face_from_images(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return face

def save_image(image, face, filename):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    for y, x in enumerate(face):
        img = img.crop((x[0]-20, x[1]-20, x[0]+x[2]+20, x[1]+x[3]+20))
        img.save(f"{filename}_{y}.jpg")

def main():
    
    list_of_filenames = get_filenames()
    
    for y, filename in enumerate(list_of_filenames):
    
        image = get_image(filename)
        face = get_face_from_images(image)
        
        if len(face) == 0:
            print(filename)
            continue
            
        save_image(image, face, f"{new_filename}_{filename}_{y}")
    
if __name__ == "__main__":
    main()