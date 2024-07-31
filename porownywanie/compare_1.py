
from os import getcwd, listdir
from sys import path
path.append(getcwd() + "\\face_comparer_master")
from face_comparer_master.train import create_model
from imageio.v2 import imread
from skimage import color
import numpy as np
import traceback
from PIL import Image

weights_path = getcwd() + "\\face_comparer_master\\experiments\\weights\\" + "tr3.last.weights"
refference_path = getcwd() + "\\faces\\refference\\"
test_path = getcwd() + "\\faces\\test\\"

def load_images():

    test_batch, test_label = [], []
    for test_image in listdir(test_path):
        #print("\n\n--" + test_image + "\n\n")
        
        test_label.append(test_image)
        tst_img = Image.open(test_path + test_image)
        tst_img = tst_img.resize((32, 32)).convert('L')
        tst_img = np.array(tst_img)[:, :, np.newaxis]
        #print(tst_img.shape)
        #print(type(tst_img))
        test_batch.append(tst_img)
        
    ref_batch, ref_label = [], []
    for ref_image in listdir(refference_path):
        #print("\n\n------" + ref_image + "\n\n")
        
        ref_label.append(ref_image)
        ref_img = Image.open(refference_path + ref_image)
        ref_img = ref_img.resize((32, 32)).convert('L')
        ref_img = np.array(ref_img)[:, :, np.newaxis]
    
        ref_batch.append(ref_img)
        
        
    #print(f"{len(test_batch)} {len(ref_batch)}")
        
    return ref_batch, test_batch, ref_label, test_label

def main():
    model = create_model()[0]
    model.load_weights(weights_path)
        
    ref_batch, test_batch, ref_label, test_label = load_images()
    
    for y, test in enumerate(test_batch):
        
        print(f"{test_label[y]} =?= {ref_label[0]}")
        preds = model.predict([ np.array( ref_batch ), np.array( [test]*len(ref_batch) ) ])
        print(sum(preds) / len(preds))
        

    input("3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        
    input()
