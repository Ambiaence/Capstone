import os
import copy
import pickle
import skimage
import numpy
import matplotlib.pyplot as plt
import math

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def get_8_by_8_connical_letters(letter):
    images = list()
    path1 = r"Training Set/font" + "1" + r"/" + letter  + r".png"
    path2 = r"Training Set/font" + "2" + r"/" + letter  + r".png"
    path3 = r"Training Set/font" + "3" + r"/" + letter  + r".png"

    letter_image = skimage.io.imread(path1)
    letter_image = down_scale_to_8_by_8_and_cut(letter_image)
    images.append(letter_image)

    letter_image = skimage.io.imread(path2)
    letter_image = down_scale_to_8_by_8_and_cut(letter_image)
    images.append(letter_image)

    letter_image = skimage.io.imread(path3)
    letter_image = down_scale_to_8_by_8_and_cut(letter_image)
    images.append(letter_image)

    return images

def get_ratio(image):
    width = image.shape[1]
    height = image.shape[0]

    ratio = width/height
    if ratio < 1:
        ratio = - (1/ratio)
        ratio = ratio+1
    else:
        ratio = ratio - 1
    
    return ratio

    
def down_scale_to_8_by_8_and_cut(letter_image):
        letter_image=cut_image(letter_image)
        letter_image = skimage.transform.resize(letter_image, (64,64), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (32,32), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (16,16), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (8,8), anti_aliasing=False)
        return letter_image

def down_scale_to_8_by_8(letter_image):
        letter_image = skimage.color.rgb2gray(letter_image[:,:,:3])
        letter_image = skimage.transform.resize(letter_image, (64,64), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (32,32), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (16,16), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (8,8), anti_aliasing=False)
        return letter_image

def cut_image(image): 
    cut_image = skimage.color.rgb2gray(image[:,:,:3])
    cut_image = skimage.util.img_as_bool(cut_image)
    cut_image = skimage.util.img_as_ubyte(cut_image)

    top = float("inf")
    bottom = float("-inf")
    left = float("inf")
    right = float("-inf")


    for row in range(0, cut_image.shape[0]):
        for column in range(0, cut_image.shape[1]):
            if cut_image[row][column] == False: 
                top = min(top, row)
                bottom = max(bottom, row)
                left = min(left, column)
                right = max(right, column)
    width = abs(left-right)
    height = abs(top-bottom)
    cut_image=cut_image[top:top+height, left:left+width] 
    return cut_image

letters = list()
letter_images = list()

position = ord("a")
front_position = [1]

for letter in range(0, 26):
    letters.append(chr(position))
    position = position+1 
print(letters)

for letter in letters:
    for font in range(1,4):
        path = r"Training Set/font" + str(font) + r"/" + letter  + r".png"
        letter_image = skimage.io.imread(path)

        letter_image=cut_image(letter_image)
        ratio = get_ratio(letter_image)
        letter_image = skimage.transform.resize(letter_image, (64,64), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (32,32), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (16,16), anti_aliasing=False)
        letter_image = skimage.transform.resize(letter_image, (8,8), anti_aliasing=False)
        letter_images.append((letter, ratio, letter_image))

    data = list()
    for letter in letter_images:
        character, ratio, image = letter
        data.append((character, ratio, image.flatten()))

def classify(image):
    if not os.path.exists("classifier.model"):
        print("No trained model")
        return None

    cfile = open('classifier.model', 'rb')    
    classifier = pickle.load(cfile)
    image_mutable = copy.deepcopy(image)
    image_mutable = image.flatten()
    return classifier.predict([image_mutable])


if os.path.exists("classifier.model"):
    print("Model already exists")
    cfile = open('classifier.model', 'rb')    
    classifier = pickle.load(cfile)

    for letter in data:
        character, ratio, image = letter
        print(character, classifier.predict([image]))
else:
    x_data = list()
    y_data = list()
    for letter in data:
        character, ratio, image = letter
        x_data.append(image)
        y_data.append(ord(character)-96)
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(x_data,y_data)

    for letter in data:
        character, ratio, image = letter
        print(character, classifier.predict([image]))

    cfile = open('classifier.model', 'ab')
    pickle.dump(classifier, cfile)
    cfile.close