import os
from shutil import copy2 as copyfile
import traceback
from pprint import pprint
import cv2
import face_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
from time import time_ns

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
image_output = path_to_faces + "\\output"
image_debug = path_to_faces + "\\debug"
directory_paths = [image_input_refference_directory, image_processed_refference_directory, image_input_test_directory, image_processed_test_directory, image_debug, image_output]

face_counter = 0
crop_offset = 10
comparison_cutoff = 0.5

use_subdivisions = True
subdivisions = [
    {'order': 1, 'type':'sparse'}, # this should be the default option
    {'order': 2, 'type':'sparse'},
    {'order': 4, 'type':'dense'},
    {'order': 6, 'type':'sparse'},
#    {'order': 2, 'type':'dense'},
]

def divide_image(image, subdiv):

    output = []
    
    image = image.numpy_view()
    height = len(image)
    width = len(image[0])
    
    sd_height = height // subdiv['order'] #subdivision height
    sd_width = width // subdiv['order']
    #print(f"{height=} {width=} {sd_height=} {sd_width=}")
    
    if subdiv['type'] == 'sparse':
    
        for x_count in range(subdiv['order']):
            for y_count in range(subdiv['order']):
            
                #print(f"{sd_height * y_count} {sd_height * (y_count+1)} {sd_width * x_count} {sd_width * (x_count+1)}")
                n_img = image[sd_height * y_count : sd_height * (y_count+1),
                            sd_width * x_count : sd_width * (x_count+1)
                ]
                
                n_img = n_img.astype(np.uint8)
                
                output.append([mp.Image(image_format=mp.ImageFormat.SRGB, data=n_img),
                        [sd_width * x_count, sd_height * y_count, sd_width, sd_height] # originX, originY, width, height
                    ])
                    
    elif subdiv['type'] == 'dense':
        #here, width should stay the same as in sparse, but starting offsets should be smaller
        
        b_height = height // (2 * subdiv['order']) # beginning height
        b_width = width // (2 * subdiv['order'])
        
        for x_count in range(2 * subdiv['order'] - 1):
            for y_count in range(2 * subdiv['order'] - 1):
            
                #print(f"{b_height * y_count} {b_height * y_count + sd_height} {b_width * x_count} {b_width * x_count + sd_width}")
                #print(f"  {b_height * y_count + sd_height - (b_height * y_count)} {b_width * x_count + sd_width - (b_width * x_count)}")
                n_img = image[b_height * y_count : b_height * y_count + sd_height,
                            b_width * x_count : b_width * x_count + sd_width
                ]
                
                n_img = n_img.astype(np.uint8)
                
                output.append([mp.Image(image_format=mp.ImageFormat.SRGB, data=n_img),
                        [b_width * x_count, b_height * y_count, sd_width, sd_height] # originX, originY, width, height
                    ])
        
    #return list of subdivided images, with their offset in relation to original coordinates
    # preferably in [[image, offset], [image, offset]] format
    return output

def get_faces():
    with FaceDetector.create_from_options(options) as detector:
    
        print("\r\n extracting refference faces\r\n")
        for file in os.listdir(image_input_refference_directory):
        
            image = mp.Image.create_from_file(image_input_refference_directory + "\\" + file)  # read image here, not in get_face, to not load same image multiple times
            
            for subdivision in subdivisions:
            
                #print(file, subdivision)
                get_face(detector, image, subdivision, image_input_refference_directory, image_processed_refference_directory, file)
            
            
        print("\r\n extracting test faces\r\n")
        for file in os.listdir(image_input_test_directory):
        
            image = mp.Image.create_from_file(image_input_test_directory + "\\" + file)
            
            for subdivision in subdivisions:
                get_face(detector, image, subdivision, image_input_test_directory, image_processed_test_directory, file)

def get_face(detector, image, subdivision, path, save_path, filename):
    global face_counter
    image_face_counter = 0

    # dziel na podobrazy
    subimages = divide_image(image, subdivision)
    
    #dla każdego podobrazu, wykryj twarze, zachowaj przesunięcia
    for subdiv_index, subimage in enumerate(subimages):
        
        #img = Image.fromarray(subimage[0].numpy_view())
        #img.save(image_debug + "\\" + f"{filename}_{subdiv_index}.{filename.split('.')[-1]}")
        result = detector.detect(subimage[0])
        
        faces = [ ( y:=x.bounding_box, [y.origin_x, y.origin_y, y.width, y.height] )[1]
                  for x in result.detections
                ]
        
        if len(faces) == 0:
            #print(f"Couldn't extract faces from {filename} subdiv {subdivision['order']}-{subdiv_index}")
            continue
        else:
            print(f"extracting faces from {filename}")
        
        for y, face in enumerate(faces):
            filnam = filename.split(".")
            face_counter += 1
            #print(filename, y)
            #print("\r\n" + filename + "\r\n")
            img = Image.fromarray(subimage[0].numpy_view())
            img = img.crop((face[0]-crop_offset, face[1]-crop_offset, face[0]+face[2]+crop_offset, face[1]+face[3]+crop_offset))
            #img = ImageOps.grayscale(img)
            img.save(save_path + "\\" + '.'.join(filnam[:-1]) + f"_subdiv{subdivision['order']}{subdivision['type'][0]}-{subdiv_index}_{image_face_counter}.{filnam[-1]}")
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
    
    output = []
    refference = [encode_face(x, image_processed_refference_directory)[0] for x in os.listdir(image_processed_refference_directory)]
    
    for test in os.listdir(image_processed_test_directory):
        
        distance = compare(
            refference,
            encode_face(test, image_processed_test_directory)[0]
            #-------------------------------------------------
        )
        output.append([test, avg:=sum(distance) / len(distance), distance])
        #print(f"filename: {test}, average:{avg}, {distance}")
        print(f"comparing: {test}")
        file.write(f"filename: {test}, average:{avg}, {distance}\r\n")
        
    return output

def move_images_similar(comparisons, file):
    
    for comparison in comparisons:
        
        ext = comparison[0].split('.')[-1]
        filename = comparison[0].split("_")[:-2]
        filename = '_'.join(filename) + '.' + ext
        print(f"copying: {filename}")
        file.write(f"Copied: {filename}")
    
        #check if anything in the comparison list is smaller than comparison cutoff
        if [1 for x in comparison[2] if x < comparison_cutoff]:
            copyfile(image_input_test_directory + "\\" + filename, image_output + "\\" + filename)
        
    
    
    

def make_folders():
    
    for x in directory_paths:
        Path(x).mkdir(parents=True, exist_ok=True)

def clear_folders():
    
    for dirr in [image_processed_refference_directory, image_processed_test_directory, image_debug]:
        for x in os.listdir(dirr):
            os.remove(dirr + "\\" + x)

def ask_get_faces():
    
    inp = input("\r\n" + "Skip extracting faces from files? (Write 'Yes' to skip): ")
    
    if inp in ['yes', 'Yes', "'Yes'", "'yes'", 'Y', 'y']:
        return False
    return True

def main():
    
    answer = ask_get_faces()
    
    make_folders()
    print("created folders")
    
    if answer:
        clear_folders()
        print("cleaned processed folders")
        
        get_faces()
        print("faces acquired")
        print(f"found {face_counter} faces")    
        input("press enter to continue")
        
    else:
        print("Skipping face extraction")
        
    with open("log.txt", "w") as file:
        file.write(f"{face_counter} faces found\r\n")
        comparisons = compare_faces(file)
        move_images_similar(comparisons, file)
    

if __name__ == "__main__":
    try:
        t1 = time_ns()
        main()
        t2 = time_ns()
        print("\r\nfinished")
        print(f"Time taken: {(t2-t1)* (10**9)}s")
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        input("error")
    