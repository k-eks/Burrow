from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import h5py
import numpy as np
import xds_tools
import meerkat_tools
import math
import glob
import fabio
import os

def filler_meerkat_average(clearedFile, originalFile):
    """Turns untrusted voxels and nans to zero and fills punched Braggs."""
    """The filling of the Braggs is done in a two dimensional manner."""
    meta = meerkat_tools.MeerkatMetaData(clearedFile)
    outfile = h5py.File("filled_" + originalFile.filename, 'w')
    outfile.create_dataset("data", (meta.shape[0],meta.shape[1],meta.shape[2]), dtype=np.float64)

    for i in range(meta.shape[2]):
        print("Using section " + str(i) + " of " + str(meta.shape[2] - 1), end="\r")
        section_punched = meerkat_tools.MeerkatSection(clearedFile, i)
        section_original = meerkat_tools.MeerkatSection(originalFile, i)
        #restoring the untrusted regions
        for x in range(meta.shape[0]):
            for y in range(meta.shape[1]):
                if math.isnan(section_original.data[x,y]):
                    section_punched.data[x,y] = 0
                elif meerkat_tools.is_untrusted(section_original.data[x,y]):
                    section_punched.data[x,y] = 0
                elif math.isnan(section_original.data[x,y]):
                    section_punched[x,y] = 0

        #searching for Braggs and fill them
        for x in range(meta.shape[0]):
            for y in range(meta.shape[1]):
                if meerkat_tools.is_hole(section_punched.data[x,y]):
                    bragg = find_hole(np.array([]), section_punched.data, x, y)
                    # after a complete set of Bragg pixels is found, the mean of
                    # surrounding pixels is used to fill the hole
                    dilated = dilation(bragg, meta.shape)
                    dilated = remove_similar_pixel(dilated, bragg)
                    mean_intensity = 0
                    for pixel in dilated:
                        mean_intensity = mean_intensity + section_punched.data[pixel.x, pixel.y]
                    mean_intensity = mean_intensity / len(dilated)
                    for pixel in bragg:
                        section_punched.data[pixel.x, pixel.y] = mean_intensity

        # writing the
        outfile['data'][:,:,i] = section_punched.data
    outfile.close()


def filler_frame_average(pathToFrames, pathToFilled):
    """Fills detector frames with the average of the surrounding pixel."""
    for imageFile in sorted(glob.glob(pathToFrames + "*.cbf"), key=xds_tools.numericalSort):
        print("Using " + str(imageFile), end="\r")
        frame = fabio.open(imageFile)
        for x in range(frame.data.shape[0]):
            for y in range(frame.data.shape[1]):
                if frame.data[x,y] == xds_tools.MASKED_BRAGG_INTENSITY:
                    bragg = find_Bragg_hole(np.array([]), frame.data, x, y)
                    dilated = dilation(bragg, frame.data.shape)
                    masked = find_masked_pixel(dilated, frame.data)
                    dilated = remove_similar_pixel(dilated, np.append(bragg, masked))
                    mean_intensity = 0
                    for pixel in dilated:
                        mean_intensity = mean_intensity + frame.data[pixel.x, pixel.y]
                    mean_intensity = mean_intensity / len(dilated)
                    for pixel in bragg:
                        frame.data[pixel.x, pixel.y] = mean_intensity
        frame.write(pathToFilled + os.path.basename(imageFile))
        del frame # freeing memory
    print("\nFilling complete!")


def find_Bragg_hole(hole, data, x, y):
    """Algorithm for recursive search of holes in a frame."""
    """Looks for all pixels connected to the masked pixel."""
    if not pixel_in_array(hole, meerkat_tools.Pixel(x,y)):
        hole = np.append(hole, meerkat_tools.Pixel(x,y))
        # straight connections
        if x > 0: # Compensate for border
            if data[x-1, y] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x-1, y)
        if x < (data.shape[0] - 1): # Compensate for border
            if data[x+1, y] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x+1, y)
        if y > 0: # Compensate for border
            if data[x, y-1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x, y-1)
        if y < (data.shape[1] - 1): # Compensate for border
            if data[x, y+1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x, y+1)
        # diagonal connections
        if x > 0 and y > 0: # Compensate for border
            if data[x-1, y-1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x-1, y-1)
        if (x < (data.shape[0] - 1)) and (y < (data.shape[1] - 1)): # Compensate for border
            if data[x+1, y+1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x+1, y+1)
        if (x < (data.shape[0] - 1)) and (y > 0): # Compensate for border
            if data[x+1, y-1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x+1, y-1)
        if (x > 0) and (y < (data.shape[1] - 1)): # Compensate for border
            if data[x-1,y+1] == xds_tools.MASKED_BRAGG_INTENSITY:
                hole = find_Bragg_hole(hole, data, x-1, y+1)
    return hole


def find_hole(hole, data, x, y):
    """Algorithm for recursive search of holes in a 2D cut."""
    """Looks for all pixels connected to the punched pixel."""
    if not pixel_in_array(hole, meerkat_tools.Pixel(x,y)):
        hole = np.append(hole, meerkat_tools.Pixel(x,y))
        # straight connections
        if x > 0: # Compensate for border
            if meerkat_tools.is_hole(data[x-1, y]):
                hole = find_hole(hole, data, x-1, y)
        if x < (data.shape[0] - 1): # Compensate for border
            if meerkat_tools.is_hole(data[x+1, y]):
                hole = find_hole(hole, data, x+1, y)
        if y > 0: # Compensate for border
            if meerkat_tools.is_hole(data[x, y-1]):
                hole = find_hole(hole, data, x, y-1)
        if y < (data.shape[1] - 1): # Compensate for border
            if meerkat_tools.is_hole(data[x, y+1]):
                hole = find_hole(hole, data, x, y+1)
        # diagonal connections
        if x > 0 and y > 0: # Compensate for border
            if meerkat_tools.is_hole(data[x-1, y-1]):
                hole = find_hole(hole, data, x-1, y-1)
        if (x < (data.shape[0] - 1)) and (y < (data.shape[1] - 1)): # Compensate for border
            if meerkat_tools.is_hole(data[x+1, y+1]):
                hole = find_hole(hole, data, x+1, y+1)
        if (x < (data.shape[0] - 1)) and (y > 0): # Compensate for border
            if meerkat_tools.is_hole(data[x+1, y-1]):
                hole = find_hole(hole, data, x+1, y-1)
        if (x > 0) and (y < (data.shape[1] - 1)): # Compensate for border
            if meerkat_tools.is_hole(data[x-1,y+1]):
                hole = find_hole(hole, data, x-1, y+1)
    return hole

def dilation(hole, shape):
    """Exentds the current hole by its next neighbours"""
    dilated = np.array([])
    for pixel in hole:
        # straight connections
        if pixel.x > 0: # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x-1, pixel.y))
        if pixel.x < (shape[0] - 1): # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x+1, pixel.y))
        if pixel.y > 0: # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x, pixel.y-1))
        if pixel.y < (shape[1] - 1): # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x, pixel.y+1))
        # diagonal conections
        if pixel.x > 0 and pixel.y > 0: # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x-1, pixel.y-1))
        if (pixel.x < (shape[0] - 1)) and (pixel.y < (shape[1] - 1)): # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x+1, pixel.y+1))
        if (pixel.x < (shape[0] - 1)) and (pixel.y > 0): # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x+1, pixel.y-1))
        if (pixel.x > 0) and (pixel.y < (shape[1] - 1)): # Compensate for border
            dilated = np.append(dilated, meerkat_tools.Pixel(pixel.x-1, pixel.y+1))
    dilated = remove_duplicate_pixel(dilated)
    return dilated

def pixel_in_array(searchArray, pixel):
    """Looks if a pixel object is in the given array"""
    result = False
    for item in searchArray:
        if item.x == pixel.x and item.y == pixel.y:
            result = True
            break
    return result

def remove_duplicate_pixel(searchArray):
    """Removes all additional pixel objects which have the same coordinates"""
    unique = np.array([])
    for item in searchArray:
        if not pixel_in_array(unique, item):
            unique = np.append(unique, item)
    return unique

def find_masked_pixel(searchArray, data):
    """Collects all pixel which belong to the masked area."""
    maskedFree = np.array([])
    for item in searchArray:
        if data[item.x, item.y] == xds_tools.SIMPLE_UNTRUSTED:
            maskedFree = np.append(maskedFree, item)
    return maskedFree


def remove_similar_pixel(searchArray, toRemove):
    """Removes all pixel objects in searchArray  which have the same coordinates as """
    """the pixels in toRemove."""
    similar = np.array([])
    for item in searchArray:
        if not pixel_in_array(toRemove, item):
            similar = np.append(similar, item)
    return similar