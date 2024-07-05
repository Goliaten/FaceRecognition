
from os import listdir, remove
from PIL import Image

dir = "faces"

for x in listdir(dir):
    print(x)
    img = Image.open(f"{dir}\\" + x)
    img = img.convert("RGB")
    remove(f"{dir}\\" + x)
    img.save(f"{dir}\\" + x.split(".")[0] + ".jpeg")
