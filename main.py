import math
import webbrowser
import imageio
from sklearn.cluster import KMeans
import numpy as np
import copy
import word
from word import open_dictionary

from word import normailze_to_frequency
from llm import spell_check
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
from down import classify
from down import down_scale_to_8_by_8
from down import get_8_by_8_connical_letters
import skimage as ski
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import data, segmentation, color, filters, io
import skimage

import os
import io
import PySimpleGUI as psg
from PIL import Image, ImageGrab

global frequency_lookup
global rank_lookup 
frequency_lookup = dict()
rank_lookup = dict()
open_dictionary(frequency_lookup, rank_lookup)

WIDTH = 700 

def sort_and_purge(boundries):
    for index, boundrie in enumerate(boundries):
        left, right = boundrie[1]
        if left < 0 or left >= 1:
            del boundries[index]
        if right < 0 or right >= 1:
            del boundries[index]
    return sorted(boundries, key=lambda tup: tup[1][0])

def refresh_letter_image(): 
    global scikit_letter_image
    top = values["-SLIDER-PARSE-TOP-"]
    bottom = values["-SLIDER-PARSE-BOTTOM-"]
    number = int(window["-LETTER-NUM-"].get())
    shape, ratio = ratio_boundries[number-1]
    scikit_letter_image = sliced_windowed_image(scikit_image, ratio)
    scikit_letter_image = return_normalize_display_letter_image(scikit_letter_image)
    skimage.io.imsave("letter_temp.png", scikit_letter_image)
    window["-LETTER-IMAGE-"].update("letter_temp.png")

def get_letter_number():
    number = int(window["-LETTER-NUM-"].get())
    return number

def update_displayed_guess():
    string = "WIP Word -> " + str(current_word.letters)
    string = str(string)
    window["-WORD-GUESS-"].update(string) 

def extant_word():
    global current_word
    try:
        current_word = current_word
        return current_word
    except:
        return word.Word(len(ratio_boundries))

def show_image(image):
    skimage.io.imshow(image)
    plt.show()

def sliced_windowed_image(image, ratios):
    left, right = ratios # grab floating point values
    length = image.shape[0]
    width = image.shape[1]
    left = math.floor(left*width)  #Scale
    right = math.floor(right*width)  #Scale
    sliced_image = copy.deepcopy(image) 
    sliced_image = sliced_image[0:length, left:right ]  # Slice important part
    return sliced_image

def draw_verticle_line(image, position):
    width = image.shape[1]
    height = math.floor(position*image.shape[0])
    rr, cc = skimage.draw.line(height, 0, height, width-1)
    image[rr, cc] = 0.5 
    return image

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

def return_normalize_display_letter_image(image):
    desired_height = 100
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_width/image_height
    desired_width =  math.floor(image_ratio*desired_height)
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
            pairs.append((left,right))
        except:
            continue

    for pair in pairs:
        left, right = pair
        rr, cc = skimage.draw.line(0, left, height-1, left)
        image[rr, cc] = 34 
        rr, cc = skimage.draw.line(0, right, height-1, right)
        image[rr, cc] = 34 
    
        for middle in range(left, right):
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

   # [psg.Radio("K-Mean Filter", "filter_radio", key="-K-MEAN-"), psg.Radio("No Filter", "filter_radio", key="-K-MEAN-")],

   [psg.Slider(range=(1,3), default_value=1,
   expand_x=True, enable_events=True,
   orientation='horizontal', key='-SCALE-')],

   [psg.Image('nowordyet.png',
   expand_x=True,
   expand_y=True,
   key="-IMAGE-")],

   [psg.Text(text='Skeletonize ->',
   font=('Arial Bold', 16),
   size=20, expand_x=True,
   justification='center'),
   psg.Button("Skeletonize")],
]

parse_letters_previous = psg.Button("<-")
parse_letters_next = psg.Button("->") 
parse_letters_border_number = psg.Text(text='1', key ="-BOUND-NUM-")
parse_letters_show_all = psg.Button("Show All") 
parse_letters_delete_current = psg.Button("Delete Current") 
parse_letters_show_all = psg.Button("Split Current") 
parse_letters_move = psg.Button("Show All") 
parse_letters_show_current = psg.Button("Show Current") 
parse_letters_image_flip = psg.Button("Switch Image")
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
   [psg.Button("Read Letter"),psg.Button("Modify Left Boundrie") , psg.Button("Modify Right Boundrie"), parse_letters_previous, parse_letters_border_number,  parse_letters_next],
   [parse_letters_delete_current, parse_letters_show_all, parse_letters_image_flip, parse_letters_show_current, parse_letters_move,]

]

tab_contents_read_letters_row_one = [
  psg.Slider(range=(1, 100), default_value=100,
   expand_x=True, enable_events=True,
   orientation='vertical', key='-SLIDER-PARSE-TOP-'),

   psg.Image('noletteryet.png',
   expand_x=False,
   expand_y=True,
   key="-LETTER-IMAGE-"),

  psg.Slider(range=(1, 100), default_value=1,
   expand_x=True, enable_events=True,
   orientation='vertical', key='-SLIDER-PARSE-BOTTOM-'),
]

tab_contents_read_letters_row_two = [
    psg.Button("Previous Letter"),
    psg.Text(text='1', key ="-LETTER-NUM-"),
    psg.Button("Next Letter"),
    psg.Button("Get Help"),
    psg.Text(text='Full Guess', key ="-WORD-GUESS-"),
    psg.Button("Get More Info"),
    psg.Radio("Extra Info", "info_radio", key="-INFO-RADIO-"),
]

tab_contents_read_letters_row_three = [
    psg.Text(text='AI Guess: ', key ="-AI-GUESS-"),
    psg.Button("Assign Letter"),
    psg.Input('User Guess', enable_events=True, key='-USER-GUESS-', font=('Arial Bold', 20), justification='left'),
]

tab_contents_read_letters = [
   tab_contents_read_letters_row_one,
   tab_contents_read_letters_row_two,
   tab_contents_read_letters_row_three,
]

tab_contents_spell_check = [[
    psg.Button("Frequency Information"),
    psg.Button("AI Help"),
    psg.Text(text="Suggested Words", key="-WORDS-")
]]

layout = [[
    psg.TabGroup([
        [psg.Tab("Grab Word", tab_contents_grab_word),
        psg.Tab("Borders", tab_contents_parse_letters),
         psg.Tab("Read Letters", tab_contents_read_letters, key = "Read Letters"),
         psg.Tab("Spell Check", tab_contents_spell_check)]
    ], key = "tabgroup", enable_events=True)]
]
      

word_bytes = io.BytesIO()
skeleton_bytes = io.BytesIO()
shape_dictionary = dict()

mutable_boundrie = "left"
word_loaded = False
is_display_skeleton = True
skeleton_created = False
window = psg.Window('English Reading Buddy',  layout, size=(1200,600), keep_on_top=True,font="Arial 12",)

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
        draw_boundries([ratio_boundries[number-1]], boundry_image)
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

    
      #if values["-K-MEAN-"] is True:
       # window_dictionary = dict()
       # area_dictionary = dict()
       # min_r = dict()
       # max_r = dict()
       # max_c = dict()
       # min_c = dict()

       # for shape in set(shape_dictionary.values()):
       #     min_r[shape] = float("inf") 
       #     max_r[shape] = float("-inf")
       #     min_c[shape] = float("inf")
       #     max_c[shape] = float("-inf")
       # 
       # for pixel, shape in shape_dictionary.items():
       #     row = pixel[0]
       #     column = pixel[1]

       #     min_r[shape] = min(row, min_r[shape])
       #     max_r[shape] = max(row, max_r[shape])

       #     min_c[shape] = min(column, min_c[shape])
       #     max_c[shape] = max(column, max_c[shape])
       # 
       # for shape in set(shape_dictionary.values()):
       #     window_dictionary[shape] = ((min_r[shape], min_c[shape]), (max_r[shape], max_c[shape]))
       #     area_dictionary[shape] = (abs(min_r[shape]-max_r[shape])*abs(max_c[shape]-max_r[shape]))

       # for shape in set(shape_dictionary.values()):
       #     print(window_dictionary[shape])
       #     print(area_dictionary[shape])

       # temp_array = list()

       # for row in area_dictionary.items():
       #     print(row[0])
       #     temp_array.append([row[1], 0])

       # X = np.array(temp_array)
       # kmeans = KMeans(n_clusters=2, random_state=10, n_init="auto").fit(X)
       # print(kmeans.labels_)

       # for shape, window_ in window_dictionary.items():
       #     if kmeans.labels_[shape-1] == 1:
       #         print(window_)
       #         windowed_display_image = skimage.util.img_as_uint(scikit_skeleton)
       #         windowed_display_image = skimage.color.gray2rgb(windowed_display_image)
       #         windowed_display_image = windowed_image(windowed_display_image, window_) 
       #         windowed_display_image = return_normalize_display_image_rgb(windowed_display_image) 
       #         plt.imsave("test.png", windowed_display_image)
       #         window["-IMAGE-SKELETON-"].update("test.png")
       #         breakpoint()
             
      #Calculate boundries for windows
      boundries = get_boundries(shape_dictionary)
      skeleton_created = True
      ratio_boundries = list()
      boundries = list(boundries.items())
      
      #Remove smaller boundries from overlaps
      for shape_r in boundries:
        for shape_c in boundries:
            
            number_one, line_one = shape_r
            number_two, line_two = shape_c

            if number_one == number_two:
                continue

            if do_lines_intersect(line_one, line_two):
                length_one = line_one[0] - line_one[1]
                length_two = line_two[0] - line_two[1]
                if (length_one > length_two):
                    boundries.remove(shape_r)
                else:
                    boundries.remove(shape_c)

      for boundrie in boundries:
        number, two_tuple = boundrie
        left, right = two_tuple
        width = scikit_skeleton.shape[1]
        ratio_left = left/width
        ratio_right = right/width
        ratio_boundries.append((number, (ratio_left, ratio_right)))

      boundries = sorted(boundries, key=lambda tup: tup[1][0])
      #ratio_boundries = sorted(ratio_boundries, key=lambda tup: tup[1][0])
      ratio_boundries =sort_and_purge(ratio_boundries)
      boundry_image = skimage.io.imread("borderdisplay.png")
      draw_boundries(ratio_boundries, boundry_image)
      skimage.io.imsave("bound_temp.png", boundry_image)
      window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")
    
   if event == "->" and skeleton_created is True:
    number = int(window["-BOUND-NUM-"].get())
    if number < len(ratio_boundries):
        window["-BOUND-NUM-"].update(number + 1)

   if event == "<-" and skeleton_created is True:
    number = int(window["-BOUND-NUM-"].get())
    if number > 1:
        window["-BOUND-NUM-"].update(number - 1)

   if (event == "<-" or event == "->") and skeleton_created is True:
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

   if event == "Show All" and skeleton_created is True:
        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_boundries(ratio_boundries, boundry_image)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")

   if event == "Split Current":
        number = int(window["-BOUND-NUM-"].get())
        boundry_image = skimage.io.imread("borderdisplay.png")
        left, right = ratio_boundries[number-1][1]
        middle = (left+right)/2
        length = right-left

        left_boundrie = (left, middle-(length*0.08))
        right_boundrie = (middle+(length*0.08), right)

        for ratio in ratio_boundries[number:]:
            number_=ratio[0]
            ratio=(number_+1, ratio[1])

        ratio_boundries[number-1] = (number, left_boundrie)
        ratio_boundries.insert(number-1, (number, right_boundrie))
        ratio_boundries = sorted(ratio_boundries, key=lambda tup: tup[1][0])


        boundry_image = skimage.io.imread("borderdisplay.png")
        draw_boundries(ratio_boundries, boundry_image)
        skimage.io.imsave("bound_temp.png", boundry_image)
        window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")


   if event == "Switch Image" and skeleton_created is True:
        if is_display_skeleton is False:
            window["-IMAGE-SKELETON-"].update("skeleton_temp.png")
        else:
            window["-IMAGE-SKELETON-"].update("word_temp.png")
        is_display_skeleton = not is_display_skeleton

   if event == "Get Help" and skeleton_created is True:
    extra_info = values["-INFO-RADIO-"]

    downscaled_letter = copy.deepcopy(scikit_letter_image)
    height = scikit_letter_image.shape[0]
    width = scikit_letter_image.shape[0]

    top = values["-SLIDER-PARSE-TOP-"]
    bottom = values["-SLIDER-PARSE-BOTTOM-"]

    top = 1 - (top-1)/100
    bottom = 1 -  (bottom-1)/100 

    top = math.floor(top*height)
    bottom = math.floor(bottom*height)

    downscaled_letter = downscaled_letter[top:bottom, 0:width]
    downscaled_letter = down_scale_to_8_by_8(downscaled_letter)


    ai_guess = classify(downscaled_letter)[0] - 1
    ai_guess = chr(ai_guess + ord("a"))

    if extra_info is True:
        fig = plt.figure(figsize=(10, 7)) 
        rows = 1
        columns = 4
        images = get_8_by_8_connical_letters(ai_guess)
        for index, image in enumerate(images):
            fig.add_subplot(rows, columns, index+1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(str("Font #" + str(index+1)))

        fig.add_subplot(rows, columns, 4)
        plt.imshow(downscaled_letter)
        plt.axis("off")
        plt.title("Current Live Letter")
        plt.show()

    window["-AI-GUESS-"].update(ai_guess)
    current_word = extant_word()
    index = get_letter_number()-1
    current_word.letters[index] = ai_guess
    update_displayed_guess()

   if event == "tabgroup" and skeleton_created:
    refresh_letter_image()

   if event == "Previous Letter" and skeleton_created is True:
    number = get_letter_number()
    if number > 1:
        window["-LETTER-NUM-"].update(number-1)
        refresh_letter_image()

   if event == "Next Letter" and skeleton_created is True:
    number = get_letter_number()
    if number < len(ratio_boundries):
        window["-LETTER-NUM-"].update(number+1)
        refresh_letter_image()


   if event == "Delete Current":
    number = int(window["-BOUND-NUM-"].get())
    boundry_image = skimage.io.imread("borderdisplay.png")
    del ratio_boundries[number-1]
    draw_boundries(ratio_boundries, boundry_image)
    skimage.io.imsave("bound_temp.png", boundry_image)
    window["-IMAGE-BOUNDRIES-"].update("bound_temp.png")

   if event == "AI Help":
    current_word = extant_word()
    words = spell_check(str(current_word.letters))
    output = "Suggested Words are, [ \n" + words + "]"
    window["-WORDS-"].update(output)


   if event == "-SLIDER-PARSE-TOP-" or event == "-SLIDER-PARSE-BOTTOM-":
        top = values["-SLIDER-PARSE-TOP-"]
        bottom = values["-SLIDER-PARSE-BOTTOM-"]

        if top <= bottom:
            top = bottom + 4 
            if top >= 100:
                top = 100
                bottom = 1
                window["-SLIDER-PARSE-TOP-"].update(top)
                window["-SLIDER-PARSE-BOTTOM-"].update(bottom)
            else:
                window["-SLIDER-PARSE-TOP-"].update(top)

        top = (top-1)/100*-1
        bottom = (bottom-1)/100*-1
        scikit_letter_image_copy = copy.deepcopy(scikit_letter_image)
        draw_verticle_line(scikit_letter_image_copy, top)
        draw_verticle_line(scikit_letter_image_copy, bottom)
        skimage.io.imsave("letter_temp.png", scikit_letter_image_copy)
        window["-LETTER-IMAGE-"].update("letter_temp.png")

   if event == "Assign Letter":
    current_word = extant_word()
    number = int(window["-LETTER-NUM-"].get())
    guess = values["-USER-GUESS-"]
    if len(guess) == 1:
        current_word.letters[number-1] = guess
        update_displayed_guess()

   if event == "Frequency Information":
    debug_words = words 
    #debug_words = "information frequency laugh info pants"
    debug_words = debug_words.split()
    displayed_words = list() 
    bar_values = list()
    fig = plt.figure(figsize = (20, 5))
    plt.ylim(0,1)
    for word_ in debug_words:
        try:
            displayed_words.append(str(word_ + "#: " + frequency_lookup[word_]))
            bar_values.append(1 - normailze_to_frequency(rank_lookup[word_]))
        except:
            foo = foo
    


    plt.figtext(0.5, 0.01, "1 Is less frequent, 0 is most frequent. Scaled logorithmically. Format is -> Word # [Number Of Occurances In Data Set]", wrap=True, horizontalalignment='center', fontsize=12)
    plt.bar(displayed_words, bar_values, color ='maroon', width = 0.3)
    plt.show()

   if event == "Get More Info":
    current_word = extant_word()
    print(current_word.letters)

    for letter in current_word.letters:
        print(letter)
        if letter == "?":
            continue
        webbrowser.open("https://en.wikipedia.org/wiki/" + letter)
        print(letter)



   if event == '-SL-':
      window['-TEXT-'].update(font=('Arial Bold', int(values['-SCALE-'])))
window.close()