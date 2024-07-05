
from os import listdir, remove
from PIL import Image

dir = "faces"
size = 64

for x in listdir(dir):
    print(x)
    img = Image.open(f"{dir}\\" + x)
    img.thumbnail((size, size))
    img = img.crop((0, 0, size, size))
    #img.show()
    remove(f"{dir}\\" + x)
    img.save(f"{dir}\\" + x.split(".")[0] + ".jpeg")
