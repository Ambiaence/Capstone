import math
import imageio
from sklearn.cluster import KMeans
import numpy as np
import copy
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
import skimage as ski
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import data, segmentation, color, filters, io
import skimage

import os
import io
import PySimpleGUI as psg
from PIL import Image, ImageGrab

layout = [
   [psg.Text(text='English Reading Buddy',
   font=('Arial Bold', 16),
   size=20, expand_x=True,
   justification='center'),
   psg.Button("Load Image")],

   [psg.Radio("Invert", "invert_radio", key="isinvert"), psg.Radio("Default", "invert_radio", key="isinvert")],

   [psg.Slider(range=(1,3), default_value=1,
   expand_x=True, enable_events=True,
   orientation='horizontal', key='-SCALE-')],

   [psg.Text(text='Skeletonize',
   font=('Arial Bold', 16),
   size=20, expand_x=True,
   justification='center'),
   psg.Button("Skeletonize")],

   [psg.Image('temp.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-")],

   [psg.Image('temp.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-SKELETON-")]
]
      
word_bytes = io.BytesIO()
skeleton_bytes = io.BytesIO()
window = psg.Window('HelloWorld', layout, size=(715,350), keep_on_top=True)
while True:
   event, values = window.read()
   print(event, values)
   if event in (None, 'Exit'):
      break
   if event == "Load Image":
      word_bytes = io.BytesIO()
      try:
         image = ImageGrab.grabclipboard().copy()
         image.save("temp.png")
      except:
         print("An image could not be grabbed. Do you have an image in the clipboard?")
         exit()
      image.save(word_bytes,format="PNG")
      window["-IMAGE-"].update(data=word_bytes.getvalue())
   if event == "Skeletonize":
      skeleton_bytes = io.BytesIO()
      scikit_image = ski.io.imread(word_bytes)[:,:,:3]
      scikit_image = skimage.color.rgb2gray(scikit_image)
      scikit_image = rescale(scikit_image, 1, anti_aliasing=True)
      scikit_image = skimage.exposure.adjust_gamma(scikit_image, 1)
      scikit_skeleton = skeletonize(invert(scikit_image), method="zhang")
      skimage.io.imsave("skeleton_temp.png", scikit_skeleton)
      window["-IMAGE-SKELETON-"].update("skeleton_temp.png")
   if event == '-SL-':
      window['-TEXT-'].update(font=('Arial Bold', int(values['-SCALE-'])))

window.close()