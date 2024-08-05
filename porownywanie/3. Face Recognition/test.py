import os
import traceback
from pprint import pprint
import cv2
import face_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
from pathlib import Path

face_detector_model = "\\".join(os.getcwd().split("\\")[:-1]) + "\\blaze_face_short_range.tflite"
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path = face_detector_model),
    running_mode = VisionRunningMode.IMAGE,
    min_detection_confidence = 0.4
    )

path_to_faces = "\\".join(os.getcwd().split("\\")[:-1]) + "\\faces"
image_input_refference_directory = path_to_faces + "\\input_refference"
image_processed_refference_directory = path_to_faces + "\\processed_refference"
image_input_test_directory = path_to_faces + "\\input_test"
image_processed_test_directory = path_to_faces + "\\processed_test"

image_1 = "trump.jpg"
image_2 = "some1.jpg"
face_counter = 0

def get_faces():
    #with FaceDetector.create_from_options(options_full) as det_lr:
    with FaceDetector.create_from_options(options) as detector:
    
        print("\r\n extracting refference faces\r\n")
        for file in os.listdir(image_input_refference_directory):
            get_face(detector, file, image_input_refference_directory, image_processed_refference_directory)
        print("\r\n extracting test faces\r\n")
        for file in os.listdir(image_input_test_directory):
            get_face(detector, file, image_input_test_directory, image_processed_test_directory)

def get_face(detector, filename, path, save_path):
    global face_counter
    image_face_counter = 0

    image = mp.Image.create_from_file(path + "\\" + filename)
    result = detector.detect(image)
    #_------------------------
    faces = [ ( y:=x.bounding_box, [y.origin_x, y.origin_y, y.width, y.height] )[1]
              for x in result.detections
            ]
    
    if len(faces) == 0:
        print(f"Couldn't extract faces from {filename}")
        return
    else:
        print(f"extracting faces from {filename}")
    
    for y, face in enumerate(faces):
        face_counter += 1
        #print(filename, y)
        #print("\r\n" + filename + "\r\n")
        img = Image.open(path + "\\" + filename)
        img = img.crop((face[0]-10, face[1]-10, face[0]+face[2]+10, face[1]+face[3]+10))
        #img = ImageOps.grayscale(img)
        img.save(save_path + "\\" + filename)
        image_face_counter += 1
    print(f" extracted total of {image_face_counter} faces")
    
    

def encode_face(filename, path):
    #print(filename)
    image = face_recognition.load_image_file(path + "\\" + filename)
    face_location = image.shape
    face_location = (0, face_location[1], face_location[0], 0)
    
    image_encoding = face_recognition.face_encodings(image, known_face_locations=[face_location])
    
    return image_encoding

def compare(batch1, batch2):
    distance = face_recognition.face_distance(batch1, batch2)
    return distance

def compare_faces(file):
    
    refference = [encode_face(x, image_processed_refference_directory)[0] for x in os.listdir(image_processed_refference_directory)]
    #refference = []
    #for x in os.listdir(image_processed_refference_directory):
    #    y = encode_face(x, image_processed_refference_directory)
    #    if len(y) > 0:
    #        refference.append(y[0])
    
    for test in os.listdir(image_processed_test_directory):
        print()
        
        distance = compare(
            refference,
            encode_face(test, image_processed_test_directory)[0]
            #-------------------------------------------------
        )
        print(f"filename: {test}, average:{sum(distance) / len(distance)}, {distance}")
        file.write(f"filename: {test}, average:{sum(distance) / len(distance)}, {distance}\r\n")

def make_clear_folders():
    
    for x in [image_input_refference_directory, image_processed_refference_directory, image_input_test_directory, image_processed_test_directory]:
        Path(x).mkdir(parents=True, exist_ok=True)
    
    for dirr in [image_processed_refference_directory, image_processed_test_directory]:
        for x in os.listdir(dirr):
            os.remove(dirr + "\\" + x)

def main():
    
    make_clear_folders()
    print("created folders and cleaned processed")
    get_faces()
    print("faces acquired")
    print(f"found {face_counter} faces")
    input("press enter to continue")
    with open("log.txt", "w") as file:
        file.write(f"{face_counter} faces found\r\n")
        compare_faces(file)
    

if __name__ == "__main__":
    try:
        main()
        input("finished")
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        input("error")
    