import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from os import getcwd
from pprint import pprint

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

mp_image = mp.Image.create_from_file(f"input\\lebron-wemby-.jpg")
#mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)


with FaceDetector.create_from_options(options) as detector:

    face_detector_result = detector.detect(mp_image)
    print("\n\n")
    pprint(face_detector_result)
    print("\n\n")
    print(dir(face_detector_result))