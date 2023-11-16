import skimage
import numpy
import matplotlib.pyplot as plt
letters = list()
letter_images = list()

position = ord("a")
front_position = [1]

for letter in range(0, 26):
    letters.append(chr(position))
    position = position+1 
print(letters)


for letter in letters:
    path = r"Training Set/font1/" + letter  + r".png"
    letter_image = skimage.io.imread(path)
    letter_image = skimage.transform.resize(letter_image, (8,8), anti_aliasing=True)
    letter_images.append(letter_image)

print("test")
for image in letter_images:
    skimage.io.imshow(image)
    plt.show()
    print("test")
    breakpoint()

