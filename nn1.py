import tensorflow as tf

from tensorflow.keras import layers, models, Input, regularizers, Sequential
from os import listdir
from PIL import Image
from numpy import asarray, array
from random import shuffle
import matplotlib.pyplot as plt

class_names = ['lebron', 'michael']
path_to_images = "obrazy\\faces"
data_split = 8, 1, 1

def load_data():

    dir_files = listdir(path_to_images)
    images, labels = [], []
    for x in dir_files:

        img = asarray(Image.open(f"{path_to_images}\\{x}")) # opens and converts file to ndarray (used in NN)
        img = img / 255.0
        lbl = class_names.index(x.split("_")[0])
        images.append(img)
        labels.append(lbl)
    
    return images, labels

def shuffle_data(images, labels):

    zipped = list(zip(images, labels))
    shuffle(zipped)
    images, labels = zip(*zipped)

    return images, labels
    
def divide_data(images, labels):
    ind1 = int( len(images) * data_split[0] / sum(data_split))
    ind2 = int( len(images) * ( data_split[0] + data_split[1]) / sum(data_split))
    
    train_images, valid_images, test_images = array(images[:ind1]), array(images[ind1:ind2]),array(images[ind2:])
    train_labels, valid_labels, test_labels = array(labels[:ind1]), array(labels[ind1:ind2]),array(labels[ind2:])
    
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def set_up_model():
    
    data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
    ])


    model = models.Sequential()
    model.add(data_augmentation)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.add(layers.Dense(2))
    
    return model


def train_model(model, train_images, train_labels, valid_images, valid_labels):

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    history = model.fit(train_images, train_labels, epochs=15,
        validation_data=(valid_images, valid_labels))
    
    return model, history

def main():

    images, labels = load_data()
    images, labels = shuffle_data(images, labels)
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = divide_data(images, labels)
    print(len(train_images), len(valid_images), len(test_images), )
    model = set_up_model()
    
    model, history = train_model(model, train_images, train_labels, valid_images, valid_labels)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc, test_loss)
    model.save("nn1.model")

if __name__ == "__main__":
    main()
