import tensorflow as tf
from numpy import asarray, array, float32
from PIL import Image
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import array


class_names = ['lebron', 'michael']
dir = "images_to_test"
crop_offset = 15
size_x, size_y = 64, 64

model_path = os.getcwd() + "\\blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.IMAGE,
    min_detection_confidence = 0.4 # 0.5 is default
    )

def convert_image_input():

    #clearing the test folder
    for img in os.listdir(dir):
        os.remove(f"{dir}\\{img}")
    
    # convert images to jpeg, because cv2 won't read anything else
    for y, image_name in enumerate(os.listdir("input")):
        img = Image.open(f"input\\{image_name}")
        img = img.convert("RGB")
        img.save(f"converted\\{y}.jpeg")
    
    #extract faces from given images in <input> directory
    for counter, image_name in enumerate(os.listdir("converted")):
        
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=array(Image.open(f'converted\\{image_name}')) )
        with FaceDetector.create_from_options(options) as detector:
            face_detector_result = detector.detect(img)
            faces = [ ( y:=x.bounding_box, [y.origin_x, y.origin_y, y.width, y.height] )[1]
                      for x in face_detector_result.detections
                    ]
        
        if len(faces) == 0:
            print(f"Couldn't extract faces from {image_name}")
            continue
        
        img = img.numpy_view()
        img = Image.fromarray(img)
        
        for face in faces:
            
            #crop face from image
            img_save = img.crop((face[0], face[1], face[0]+face[2], face[1]+face[3]))
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
    
    for x in os.listdir(dir):
        
        img = asarray(Image.open(f"{dir}\\{x}")) # opens and converts file to ndarray (used in NN)
        img = img / 255.0
        imgs.append(img)
    return array(imgs)

def main():
    
    for x in ["input", "converted", "images_to_test"]:
        try:
            os.mkdir(x)
            print(f"Directory {x} missing, creating")
        except:
            print(f"Directory {x} present")

    model = tf.keras.models.load_model("nn1.model")

    img = get_all_data()
    prediction = model.predict(img)
    
    print(class_names)
    print(prediction)

if __name__ == "__main__":
    convert_image_input()
    main()
