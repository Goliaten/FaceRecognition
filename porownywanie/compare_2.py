
from os import getcwd, listdir
from sys import path
path.append(getcwd() + "\\face_comparer_master")
from face_comparer_master.train import create_model
from imageio.v2 import imread
from skimage import color
import numpy as np
import traceback
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = getcwd() + "\\blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.IMAGE
    )

weights_path = getcwd() + "\\face_comparer_master\\experiments\\weights\\" + "tr5.100.last.weights100"
refference_path = getcwd() + "\\faces\\input_refference\\"
test_path = getcwd() + "\\faces\\input_test\\"

def load_images():

    test_batch, test_label = [], []
    for test_image in listdir(test_path):
        
        test_label.append(test_image)
        tst_img = Image.open(test_path + test_image)
        tst_img = tst_img.convert('L')
        #tst_img = tst_img.resize((32, 32)).convert('L')
        #tst_img = np.array(tst_img)[:, :, np.newaxis]
        #print(tst_img.shape)
        #print(type(tst_img))
        test_batch.append(tst_img)
        
    ref_batch, ref_label = [], []
    for ref_image in listdir(refference_path):
        
        ref_label.append(ref_image)
        ref_img = Image.open(refference_path + ref_image)
        ref_img = ref_img.convert('L')
        #ref_img = ref_img.resize((32, 32)).convert('L')
        #ref_img = np.array(ref_img)[:, :, np.newaxis]
    
        ref_batch.append(ref_img)
        
    return ref_batch, test_batch, ref_label, test_label

def make_folders():
    pass
    
def clear_processed_folders():
    pass

def load_model():
    model = create_model()[0]
    model.load_weights(weights_path)
    return model

def extract_faces(images, path):
    with FaceDetector.create_from_options(options) as detector:
        
        for image in images:
            mp_image= mp.Image(format=mp.ImageFormat.SRGB, data=np.asarray(image))
            face_detector_result = detector.detect(mp_image)
            print("\n\n")
            pprint(face_detector_result)
            print("\n\n")
            print(dir(face_detector_result))
            input("waiting")
        print("finished")

def process_images():
    make_folders()
    clear_processed_folders()
    
    ref_batch, test_batch, ref_label, test_label = load_images()
    ref_batch = extract_faces(ref_batch, refference_path)
    test_batch = extract_faces(test_batch, test_path)
    
    return ref_batch, test_batch, ref_label, test_label

def predict():
    pass


def main():
    #model = load_model()    
    ref_batch, test_batch, ref_label, test_label = process_images()
    predict()
    
    input("3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        
    input()
