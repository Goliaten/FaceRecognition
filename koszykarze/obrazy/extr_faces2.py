from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import array
import os
import sys
from pprint import pprint


dirs = ["lebron james", "michael jordan"]

model_path = os.getcwd() + "\\blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.IMAGE,
    min_detection_confidence = 0.5 # 0.5 is default
    )

def get_filenames(dir):
    fns = os.listdir(dir)
    return fns
    
def get_image(dir, filename):
    
    img = array( Image.open(f'{dir}\\{filename}') )
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

def get_face_from_images(detector, image):

    face_detector_result = detector.detect(image)
    #pprint(face_detector_result)
    # extracts face location from detector output structure
    out = [ ( y:=x.bounding_box, [y.origin_x, y.origin_y, y.width, y.height] )[1]
            for x in face_detector_result.detections
          ]
    #pprint(out)
    return out

def save_image(image, face, filename):
    
    img = image.numpy_view()
    img = Image.fromarray(img)
    
    for y, x in enumerate(face):
        img = img.crop((x[0], x[1], x[0]+x[2], x[1]+x[3]))
        img.save(f"scipt_output_face\\{filename}_{y}.jpg")

def main():
    
    with FaceDetector.create_from_options(options) as detector:
        for dir in dirs:
            list_of_filenames = get_filenames(dir)
            new_filename = dir.split('_')[0]
            
            for y, filename in enumerate(list_of_filenames):
            
                image = get_image(dir, filename)
                faces = get_face_from_images(detector, image)
                
                if len(faces) == 0:
                    print(filename)
                    continue
                
                save_image(image, faces, f"{new_filename}_{filename}_{y}")
    
if __name__ == "__main__":
    main()