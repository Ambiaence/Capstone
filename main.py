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

WIDTH = 700 


def does_segment_contain_point(segment, point):
    s1 = segment[0]
    s2 = segment[1]
    p = point
    if p > s1 and p < s2:
        return True
    return False

def do_lines_intersect(line_one, line_two):
    a1 = line_one[0]
    a2 = line_one[1]
    b1 = line_two[0]
    b2 = line_two[1]

    does_contain = does_segment_contain_point

    if does_contain(line_one, b1) or does_contain(line_one, b2):
        return True

    if does_contain(line_two, a1) or does_contain(line_two, a2):
        return True
    
    return False

def draw_borders(image, array_of_borders):
    width = image.shape[1]
    height = image.shape[0]
    slices = list()
    for border in array_of_borders:
        slice = math.floor(border*width)
        slices.append(slice)
    
    for slice in slices:
        rr, cc = skimage.draw.line(0, slice, height-1, slice)
        image[rr, cc] = 34 
    

def return_normalize_display_image(image):
    desired_width = WIDTH
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_height/image_width
    desired_height =  math.floor(image_ratio*desired_width)
    desired_image = skimage.transform.resize(image, (desired_height, desired_width), anti_aliasing=True) 
    desired_image = skimage.util.img_as_ubyte(desired_image)
    return desired_image

def get_boundries(shapes_lookup: dict):
    mins = dict()
    maxs = dict()
    boundries = dict()

    for pixel, shape in shapes_lookup.items():
        x = pixel[1]
        if shape not in mins:
            mins[shape] = float("inf") 
            maxs[shape] = -1 
            continue

        maxs[shape] = max(x, maxs[shape])
        mins[shape] = min(x, mins[shape])

    for shape in shapes_lookup.values():
        boundries[shape] = (mins[shape], maxs[shape])

    return boundries
    

def parse_shapes(image, shape_lookup):
   traversed = set()
   shape_number=0
   for r in range(image.shape[0]):
    for c in range(image.shape[1]):
        starting_pixel = (r,c)
        if starting_pixel in traversed:
            continue

        if image[r][c] == False:
            traversed.add(starting_pixel)
            continue

        shape_number = shape_number + 1
        to_search = set()
        traversed.add(starting_pixel)
        to_search.add(starting_pixel)
        shape_lookup[starting_pixel] = shape_number

        while to_search:
            focus_pixel = to_search.pop()
            for pixel in surrounding_pixels(focus_pixel):
                i, j = pixel
                if pixel in traversed:
                    continue
                traversed.add(pixel)
                if image[i][j] == True:
                    to_search.add(pixel)
                    shape_lookup[pixel] = shape_number

def surrounding_pixels(pixel):
    r, c = pixel
    pixels = list()
    for i in range(-1, 2):
        for j in range(-1,2):
            if i == 0 and j == 0:
                continue

            if r + i < 0 or c + j < 0:
                continue

            pixels.append(tuple([r + i, c + j]))
    return pixels


tab_contents_grab_word = [
   [psg.Text(text='English Reading Buddy',
   font=('Arial Bold', 16),
   size=20, expand_x=True,
   justification='center'),
   psg.Button("Load Image")],

   [psg.Radio("Invert", "invert_radio", key="isinvert"), psg.Radio("Default", "invert_radio", key="isinvert")],

   [psg.Slider(range=(1,3), default_value=1,
   expand_x=True, enable_events=True,
   orientation='horizontal', key='-SCALE-')],

   [psg.Image('nowordyet.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-")],

   [psg.Text(text='Skeletonize',
   font=('Arial Bold', 16),
   size=20, expand_x=True,
   justification='center'),
   psg.Button("Skeletonize")],

]

parse_letters_next = psg.Button("Next") 
parse_letters_previous = psg.Button("Previous") 
parse_letters_insert = psg.Button("Insert After") 
parse_letters_previous = psg.Button("Previous") 
parse_letters_show_all = psg.Button("Show All") 
parse_letters_border_number = psg.Text(text='1')
parse_letters_border_slider = psg.Slider(range=(1,315), default_value=1,
   expand_x=True, enable_events=True,
   orientation='horizontal', key='-SCALE-')

tab_contents_parse_letters = [
   [psg.Radio("Filter", "invert_radio", key="isinvert"), psg.Radio("No Filter", "invert_radio", key="isinvert")],

   [psg.Image('nowordyet.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-SKELETON-")],

   [psg.Image('borderdisplay.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-BOUNDRIES-")],

   [parse_letters_border_slider],
   [parse_letters_previous, parse_letters_border_number, parse_letters_next, parse_letters_insert, parse_letters_show_all]
]

layout = [[
    psg.TabGroup([
        [psg.Tab("Grab Word", tab_contents_grab_word),
        psg.Tab("Figure Out Letters ", tab_contents_parse_letters) ]
    ], key = "tabgroup")]
]
      
word_bytes = io.BytesIO()
skeleton_bytes = io.BytesIO()
shape_dictionary = dict()

word_loaded = False
skeleton_created = False
window = psg.Window('HelloWorld', layout, size=(715,WIDTH), keep_on_top=True)

word_bytes = io.BytesIO()

while True:
   event, values = window.read()
   print(event, values)
   if event in (None, 'Exit'):
      break
   if event == "Load Image":
      word_bytes = io.BytesIO()
      try:
         image = ImageGrab.grabclipboard().copy()
         if type(image) is list:
            raise Exception("List not image")
      except:
         print("An image could not be grabbed. Do you have an image in the clipboard?")
         window["-IMAGE-"].update("noimageinclipboard.png")
         continue

      image.save(word_bytes, format="PNG")
      scikit_image = ski.io.imread(word_bytes)[:,:,:3]
      display_image = return_normalize_display_image(scikit_image)
      skimage.io.imsave("word_temp.png", display_image)
      window["-IMAGE-"].update("word_temp.png")
      word_loaded = True

   if event == "Skeletonize" and word_loaded is True:
      skeleton_bytes = io.BytesIO()
      scikit_gray_image = skimage.color.rgb2gray(scikit_image)
      scikit_skeleton = skeletonize(invert(scikit_gray_image), method="zhang")
      display_image = skimage.util.img_as_ubyte(scikit_skeleton)
      display_image = return_normalize_display_image(display_image) 
      skimage.io.imsave("skeleton_temp.png", display_image)
      window["-IMAGE-SKELETON-"].update("skeleton_temp.png")
      window['tabgroup'].Widget.select(1)
      shape_dictionary = dict()
      parse_shapes(scikit_skeleton, shape_dictionary)
      boundries = get_boundries(shape_dictionary)
      skeleton_created = True
      ratio_boundries = list()

      boundries = list(boundries.items())

      for shape_r in boundries:
        for shape_c in boundries:
            if shape_c[0] == shape_r[0]:
                continue
            
            line_one = shape_r[1]
            line_two = shape_c[1]
            if do_lines_intersect(line_one, line_two):
                length_one = line_one[0] - line_one[1]
                length_two = line_two[0] - line_two[1]
                if (length_one < length_one):
                    boundries.remove(shape_r)
                else:
                    boundries.remove(shape_c)
    
      boundries = sorted(boundries, key=lambda tup: tup[1][0])
    
      for first, second in zip(boundries, boundries[1:]):
        first_shape = first[0]
        second_shape = second[0]
        left_bound = first[1][1]
        right_bound = second[1][0]

        width_of_skeleton = scikit_skeleton.shape[1]
        space_bound = (left_bound + right_bound)/2 #Average
        space_bound = space_bound/width_of_skeleton #Normalize

        print(first_shape, second_shape, left_bound, right_bound, space_bound)
        ratio_boundries.append(space_bound)

      boundry_image = skimage.io.imread("borderdisplay.png")
      draw_borders(boundry_image, ratio_boundries)
      skimage.io.imsave("bound_temp.png", boundry_image)
      window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")

   if event == '-SL-':
      window['-TEXT-'].update(font=('Arial Bold', int(values['-SCALE-'])))



window.close()