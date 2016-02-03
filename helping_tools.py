from __future__ import print_function # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import os
import os.path
import glob
import time
import fabio
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def find_named_folders(rootPath, namedFolder):
    """Searches for all subdirectories which are called <namedFolder> and retruns the absolute path.
    Useful for finding all frame-directories.
    rootPath ... string path from where to start the search into all subfolders
    namedFolder ... string name of folder for which to look out
    returns list[string] of all found absolute paths to the folders which are called <namedFolder>
    """
    folderPaths = []
    print("Looking for folder paths...")
    print("Working... (this might take a while, you can go and fetch a cup of tea)")
    for path, folders, files in os.walk(rootPath): # os.walk function requires the other variables
    # found no function to just iterate over directories
        for name in folders:
            if namedFolder in name:
                folderPaths.append(os.path.join(path, name))
    print("Found %s folder paths!" % len(folderPaths))
    return folderPaths


def timestamp():
    """Generates the current timestamp.
    returns string the current time stamp in the format of YYMMDD-HHMMSS
    """
    return time.strftime("%y%m%d-%H%M%S")


def cbf_to_png(pathToFrames, addFileName=True):
    """Turns several cbf files in a folder into png's. Useful for printing!
    pathToFrames ... string path where to look for the frames
    addFileName ... bool wheter the file name should be printed onto the image or not
    """
    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf",25)
    for fileName in glob.glob(os.path.join(pathToFrames, "*.cbf")):
        frame = fabio.open(fileName)
        image = Image.fromarray(f.data, 'RGB')
        fileName += ".png"
        if addFileName:
            draw = ImageDraw.Draw(i)
            draw.text((0, 0), filename, (255,0,0), font=font)
        image.save(fileName)
        print("Written %s" % fileName)