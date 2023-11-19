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

def show_image(image):
    skimage.io.imshow(image)
    plt.show()

def windowed_image(image, window):
    r1 = window[0][0]
    c1 = window[0][1]
    r2 = window[1][0]
    c2 = window[1][1]
    windowed_image = copy.deepcopy(image) 
    number_of_rows = abs(r1-r2) + 1 
    number_of_columns = abs(c1-c2) + 1 
    window_extent = (number_of_rows, number_of_columns)
    rr, cc = skimage.draw.rectangle(start=(r1, c1), extent=(number_of_rows,number_of_columns))
    windowed_image[(rr, cc)] =  [50, 168, 113]
    return windowed_image

    

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
        try:
            slice = math.floor(border*width)
            slices.append(slice)
        except:
            print()
    
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

def return_normalize_display_image_rgb(image):
    desired_width = WIDTH
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_height/image_width
    desired_height =  math.floor(image_ratio*desired_width)
    show_image(image)
    image = skimage.util.img_as_ubyte(image)
    show_image(image)
    skimage.io.imsave("temp.bmp", image )
    show_image(image)
    desired_image = Image.open("temp.bmp") 
    #desired_image = sdkimage.transform.resize(image, (desired_height, desired_width), preserve_range=True,anti_aliasing=False) 
    desired_image.resize((desired_width, desired_height))
    show_image(desired_image)
    return desired_image

def draw_boundries(boundires, image):
    width = image.shape[1]
    height = image.shape[0]

    pairs = list()

    for boundrie in boundires:
        number, two_tuple = boundrie
        left, right = two_tuple
        try:
            left = math.floor(left*width)
            right = math.floor(right*width)
        except:
            continue
        pairs.append((left,right))

    for pair in pairs:
        left, right = pair
        rr, cc = skimage.draw.line(0, left, height-1, left)
        image[rr, cc] = 34 
        rr, cc = skimage.draw.line(0, right, height-1, right)
        image[rr, cc] = 34 
    
        for middle in range(left+1, right-1):
            rr, cc = skimage.draw.line(0, middle, height-1, middle)
            image[rr, cc] = 88  

    


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
                try:
                    if image[i][j] == True:
                        to_search.add(pixel)
                        shape_lookup[pixel] = shape_number
                except:
                    print()


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

   [psg.Radio("K-Mean Filter", "filter_radio", key="-K-MEAN-"), psg.Radio("No Filter", "filter_radio", key="-K-MEAN-")],

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
parse_letters_show_all = psg.Button("Split Current") 
parse_letters_move = psg.Button("Move To Slider") 
parse_letters_show_current = psg.Button("Show Current") 
parse_letters_image_flip = psg.Button("Switch Image")
parse_letters_border_number = psg.Text(text='1', key ="-BOUND-NUM-")
parse_letters_border_slider = psg.Slider(range=(1,315), default_value=1,
   expand_x=True, enable_events=True,
   orientation='horizontal', key='-BOUND-SCALE-')

tab_contents_parse_letters = [
   [psg.Image('nowordyet.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-SKELETON-")],

   [psg.Image('borderdisplay.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-BOUNDRIES-")],

   [parse_letters_border_slider],
   [psg.Button("Read Letter")],
   [parse_letters_previous, parse_letters_border_number, parse_letters_next, parse_letters_insert, parse_letters_show_all, parse_letters_image_flip, parse_letters_show_current, parse_letters_move]
]

tab_contents_read_letters_row_one = [
  psg.Slider(range=(10, 30), default_value=12,
   expand_x=True, enable_events=True,
   orientation='vertical', key='-SLIDER-PARSE-TOP-'),

   psg.Image('noletteryet.png',
   expand_x=True,
   expand_y=True,
   key="-LETTER-IMAGE-"),

  psg.Slider(range=(1, 100), default_value=12,
   expand_x=True, enable_events=True,
   orientation='vertical', key='-SLIDER-PARSE-BOTTOM-'),
]

tab_contents_read_letters = [
   tab_contents_read_letters_row_one,
   [psg.Button("Modify Left Boundrie"), psg.Button("Modify Right Boundrie")]
]
layout = [[
    psg.TabGroup([
        [psg.Tab("Grab Word", tab_contents_grab_word),
        psg.Tab("Borders", tab_contents_parse_letters),
         psg.Tab("Read Letters", tab_contents_read_letters)]
    ], key = "tabgroup")]
]
      
word_bytes = io.BytesIO()
skeleton_bytes = io.BytesIO()
shape_dictionary = dict()

mutable_boundrie = "left"
word_loaded = False
is_display_skeleton = True
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
    
   if event == "-BOUND-SCALE-":
        new_position = values["-BOUND-SCALE-"]/315
        number = int(window["-BOUND-NUM-"].get())
        boundrie_pair = ratio_boundries[number-1][1]
        left, right = boundrie_pair
        if mutable_boundrie == "left":
            left = new_position
        else:
            right = new_position
        new_boundrie = (left, right)
        ratio_boundries[number-1] = (number, new_boundrie)

        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_boundries(ratio_boundries, boundry_image)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")

   if event == "Skeletonize" and word_loaded is True:
      #Create Skeleton
      skeleton_bytes = io.BytesIO() 
      scikit_gray_image = skimage.color.rgb2gray(scikit_image)
      scikit_skeleton = skeletonize(invert(scikit_gray_image), method="zhang")

      #Display Skeleton
      display_image = skimage.util.img_as_ubyte(scikit_skeleton) 
      display_image = return_normalize_display_image(display_image) 
      skimage.io.imsave("skeleton_temp.png", display_image)
      window["-IMAGE-SKELETON-"].update("skeleton_temp.png")

      #Switch Tab
      window['tabgroup'].Widget.select(1)

      #Parse Shapes
      shape_dictionary = dict()
      parse_shapes(scikit_skeleton, shape_dictionary)

      if values["-K-MEAN-"] is True:
        window_dictionary = dict()
        area_dictionary = dict()
        min_r = dict()
        max_r = dict()
        max_c = dict()
        min_c = dict()

        for shape in set(shape_dictionary.values()):
            min_r[shape] = float("inf") 
            max_r[shape] = float("-inf")
            min_c[shape] = float("inf")
            max_c[shape] = float("-inf")
        
        for pixel, shape in shape_dictionary.items():
            row = pixel[0]
            column = pixel[1]

            min_r[shape] = min(row, min_r[shape])
            max_r[shape] = max(row, max_r[shape])

            min_c[shape] = min(column, min_c[shape])
            max_c[shape] = max(column, max_c[shape])
        
        for shape in set(shape_dictionary.values()):
            window_dictionary[shape] = ((min_r[shape], min_c[shape]), (max_r[shape], max_c[shape]))
            area_dictionary[shape] = (abs(min_r[shape]-max_r[shape])*abs(max_c[shape]-max_r[shape]))

        for shape in set(shape_dictionary.values()):
            print(window_dictionary[shape])
            print(area_dictionary[shape])

        temp_array = list()

        for row in area_dictionary.items():
            print(row[0])
            temp_array.append([row[1], 0])

        X = np.array(temp_array)
        kmeans = KMeans(n_clusters=2, random_state=10, n_init="auto").fit(X)
        print(kmeans.labels_)

        for shape, window_ in window_dictionary.items():
            if kmeans.labels_[shape-1] == 1:
                print(window_)
                windowed_display_image = skimage.util.img_as_uint(scikit_skeleton)
                windowed_display_image = skimage.color.gray2rgb(windowed_display_image)
                windowed_display_image = windowed_image(windowed_display_image, window_) 
                windowed_display_image = return_normalize_display_image_rgb(windowed_display_image) 
                plt.imsave("test.png", windowed_display_image)
                window["-IMAGE-SKELETON-"].update("test.png")
                breakpoint()
            
             
      #Calculate boundries for windows
      boundries = get_boundries(shape_dictionary)
      skeleton_created = True
      ratio_boundries = list()
      boundries = list(boundries.items())

      for boundrie in boundries:
        number, two_tuple = boundrie
        left, right = two_tuple
        width = scikit_skeleton.shape[1]
        ratio_left = left/width
        ratio_right = right/width
        ratio_boundries.append((number, (ratio_left, ratio_right)))


      #Remove smaller boundries from overlaps
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

      boundry_image = skimage.io.imread("borderdisplay.png")
      draw_boundries(ratio_boundries, boundry_image)
      skimage.io.imsave("bound_temp.png", boundry_image)
      window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")
    
   if event == "Next" and skeleton_created is True:
    number = int(window["-BOUND-NUM-"].get())
    if number < len(ratio_boundries):
        window["-BOUND-NUM-"].update(number + 1)

   if event == "Previous" and skeleton_created is True:
    number = int(window["-BOUND-NUM-"].get())
    if number > 1:
        window["-BOUND-NUM-"].update(number - 1)

   if (event == "Previous" or event == "Next") and skeleton_created is True:
        number = int(window["-BOUND-NUM-"].get())
        if mutable_boundrie == "left":
            position = ratio_boundries[number-1][1][0]
        else:
            position = ratio_boundries[number-1][1][1]

        adjusted_position = math.ceil(position*315)
        window["-BOUND-SCALE-"].update(adjusted_position)
    
   if event == "Modify Left Boundrie" and skeleton_created:
    mutable_boundrie = "left"
    number = int(window["-BOUND-NUM-"].get())
    new_position = math.ceil(ratio_boundries[number-1][1][0]*315)
    window["-BOUND-SCALE-"].update(new_position)

   if event == "Modify Right Boundrie" and skeleton_created:
    mutable_boundrie = "right"
    number = int(window["-BOUND-NUM-"].get())
    new_position = math.ceil(ratio_boundries[number-1][1][1]*315)
    window["-BOUND-SCALE-"].update(new_position)

   if event == "Show Current" and skeleton_created is True:
        number = int(window["-BOUND-NUM-"].get())
        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_boundries([ratio_boundries[number-1]], boundry_image)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")
        left = ratio_boundries[number-1][1][0]
        window["-BOUND-SCALE-"].update(math.ceil(left*315))

   if event == "Insert After" and skeleton_created is True:
        number = int(window["-BOUND-NUM-"].get())
        ratio_boundries.insert(number, values["-BOUND-SCALE-"]/315)
        ratio_boundries = sorted(ratio_boundries)

   if event == "Move To Slider" and skeleton_created is True:
        number = int(window["-BOUND-NUM-"].get())
        ratio_boundries[number-1] = values["-BOUND-SCALE-"]/315
        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_borders(boundry_image, [ratio_boundries[number-1]])
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")
        window["-BOUND-SCALE-"].update(math.ceil(ratio_boundries[number-1]*315))
        aatio_boundries = sorted(ratio_boundries)
    
   if event == "Split Current":
        number = int(window["-BOUND-NUM-"].get())
        boundry_image = skimage.io.imread("borderdisplay.png")
        left, right = ratio_boundries[number-1][1]
        middle = (left+right)/2
        length = right-left

        left_boundrie = (left, middle-(length*0.05))
        right_boundrie = (middle+(length*0.05), right)

        for ratio in ratio_boundries[number:]:
            number_=ratio[0]
            ratio=(number_+1, ratio[1])

        ratio_boundries[number-1] = (number, left_boundrie)
        ratio_boundries.insert(number-1, (number, right_boundrie))


        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_boundries(ratio_boundries, boundry_image)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")


   if event == "Show All" and skeleton_created is True:
        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_borders(boundry_image, ratio_boundries)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")

   if event == "Switch Image" and skeleton_created is True:
        if is_display_skeleton is False:
            window["-IMAGE-SKELETON-"].update("skeleton_temp.png")
        else:
            window["-IMAGE-SKELETON-"].update("word_temp.png")
        is_display_skeleton = not is_display_skeleton

   if event == '-SL-':
      window['-TEXT-'].update(font=('Arial Bold', int(values['-SCALE-'])))
window.close()