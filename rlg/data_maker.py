from PIL import Image, ImageDraw
from numpy.random import choice, random
import numpy as np
from PIL.ImageColor import getrgb
from os.path import normpath
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorHelper:
    '''Class for mapping from colormap to rgb'''
    
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def generate_label(features,color_index,outline):
    """Maps every feature value to the range 0-1"""
    x = (features[0]-25)/50
    y = (features[1]-25)/50
    shape = features[2]/2
    size = features[3]/25
    color = color_index
    return (x,y,shape,size,color,outline)

def generate_image(discrete, color):
    '''
    Generate an Image
    
    params:
    discrete (bool): if False, position and size will be categorical
    color (MplColorHelper): used for mapping 1D value to rgb
    '''
    im = Image.new('RGB', (100, 100), (255,255,255))
    draw = ImageDraw.Draw(im)

    vector_repr = np.zeros(7, dtype='object') #posx, posy, shape, size, c1, c2, c3

    # 'circle', 'rectangle', 'triangle'
    vector_repr[2] = choice([0, 1, 2])
    shapes = np.array([draw.ellipse, draw.rectangle, draw.regular_polygon])

    # change here for with or without outline
    outline = 1


    if discrete:
        # center position
        vector_repr[0], vector_repr[1] = (100,100)
        # size/2 in pixels
        vector_repr[3] = 25
        # one of three colors
        vector_repr[4:] = getrgb(choice(['green', 'blue', 'red']))


    else:
        # random position
        vector_repr[0], vector_repr[1] = (25+random()*50, 25+random()*50)
        # 25 <= size <= 50 in pixels
        vector_repr[3] = 5+random()*20
        # random color
        color_index = random()#*1000 #  so that we can use a single number as label, not 3 (RGB)
        vector_repr[4:] = color.get_rgb(color_index)[:3]
        vector_repr[4:] *= 255
        vector_repr[4:] = [int(x) for x in vector_repr[4:]]

        # if uncomment, images will be generated without color
        #vector_repr[4:] = [255,255,255]
        label = generate_label(vector_repr,color_index,outline)
    if vector_repr[2]==2:
        shapes[int(vector_repr[2])]((vector_repr[0],vector_repr[1],vector_repr[3]),3,fill=tuple(vector_repr[4:]),outline=outline)
    else:
        xy = ((vector_repr[0]-vector_repr[3],vector_repr[1]-vector_repr[3]), (vector_repr[0]+vector_repr[3],vector_repr[1]+vector_repr[3]))
        shapes[int(vector_repr[2])](xy=xy, fill=tuple(vector_repr[4:]),outline=outline)
        
    # save image
    with open(normpath('/home/ui/Documents/DRL project/hierarchical_reference_game/data/' + ('discrete/' if discrete else 'continuousTestdata/')+str(label)[1:-1]+'.jpg'), 'a+') as file:
        im.save(file, quality=100)


# define the color chart between 2 and 10 using the 'autumn_r' colormap, so
#   y <= 2  is yellow
#   y >= 10 is red
#   2 < y < 10 is between from yellow to red, according to its value
COL = MplColorHelper('gist_ncar', 0, 1)

if __name__ == '__main__':
    n, discrete = 25000, 1
    ren = range(n)
    for i in tqdm(ren):
        generate_image(discrete==False, color=COL)

