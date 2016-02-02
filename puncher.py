from __future__ import print_function # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import os.path
import fabio
import xds_tools
import meerkat_tools
import glob


def punch_Bragg(bgFilePath, sampleFrame, pathToFrames, pathToPunched, maskOption, hotPixel):
    """Masks all  Bragg peaks using XDS BGK files."""
    frameShape = sampleFrame.data.shape
    # generating the background data
    bgType = os.path.splitext(bgFilePath)[1]
    if bgType == ".h5":
        bg = xds_tools.bgFromH5(bgFilePath, frameShape)
    else:
        raise TypeError("Background file not supported!")

    # punching, the sorting function is used for a nice print out
    for imageFile in sorted(glob.glob(pathToFrames + "*.cbf"), key=xds_tools.numericalSort):
        print("Punching " + str(imageFile), end="\r")
        frame = fabio.open(imageFile)
        punched = frame.data - bg
        for x in range(frameShape[0]):
            for y in range(frameShape[1]):
                if maskOption == xds_tools.FULLMASK:
                    # this option enables the distinction between different detector defects
                    if frame.data[x, y] == xds_tools.UNTRUSTED:
                        punched[x,y] = xds_tools.UNTRUSTED
                    elif frame.data[x, y] == xds_tools.DEFECTIVE:
                        punched[x,y] = xds_tools.DEFECTIVE
                    elif punched[x,y] >= 0:
                        punched[x,y] = xds_tools.MASKED_BRAGG_INTENSITY
                    else:
                        punched[x,y] = frame.data[x,y]
                elif maskOption == xds_tools.SIMPLEMASK:
                    # using the simple mask
                    if frame.data[x, y] == xds_tools.SIMPLE_UNTRUSTED:
                        punched[x, y] = xds_tools.SIMPLE_UNTRUSTED
                    elif punched[x,y] >= 0:
                        punched[x,y] = xds_tools.MASKED_BRAGG_INTENSITY
                    else:
                        punched[x, y] = frame.data[x, y]
                else:
                    raise TypeError("Specified masking option \"%s\" does not exist!" % maskOption)
        frame.data = punched
        # punching of the hot pixel
        for p in hotPixel:
            frame.data[p.x, p.y] = xds_tools.MASKED_BRAGG_INTENSITY
        frame.write(pathToPunched + os.path.basename(imageFile))
        del frame # freeing memory
    print("\nPunching complete!")


def find_hot_pixel(hotFilePath, hotThreshold):
    """Reads out all pixel in the given frame which are above the given threshold."""
    hotFrame = fabio.open(hotFilePath)
    hotPixel = []
    print("looking for hot pixel...")
    for x in range(hotFrame.data.shape[0]):
        for y in range(hotFrame.data.shape[1]):
            print("Searching at x = %i y = %i          " %(x,y ), end="\r")
            if hotFrame.data[x, y] > hotThreshold:
                hotPixel.append(meerkat_tools.Pixel(x, y))
    # print out
    if len(hotPixel) > 0:
        print("\nFound hot pixel:")
        for p in hotPixel:
            print(p)
    else:
        print("\nNo hot pixel were found!")
    return hotPixel