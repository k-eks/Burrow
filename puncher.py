from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import os.path
import fabio
import xds_tools
import meerkat_tools
import helping_tools
import cbf_tools
import glob


MASKED_BRAGG_INTENSITY = -9999999


def punch_Bragg(XDSPunchFilePath, sampleFrame, pathToFrames, frameNameTemplate, pathToPunched, punchedPrefix, maskedIntensity=MASKED_BRAGG_INTENSITY):
    """
    Masks all  Bragg peaks using XDS BGK files. For an unkown reason, when XDS's BKGPIX.cbf is substracted from a raw frame, then all positive intensities represent Bragg peaks.
    XDSPunchFilePath ... string location of the BKGPIX file create by XDS. NOTE: fabio can not read the that file by default, I used ALBULA to convert it to an h5 file!
    sampleFrame ... fabio.frame some frame from which the dimensions can be read.
    pathToFrames ... string location of the folder which contains the frames.
    frameNameTemplate ... string example of how the frames are named, allows for percent substitution.
    pathToPunched ... string location where the processed frames should be saved.
    punchedPrefix ... string prefix that should be put in front of the processed frames.
    maskedIntensity ... int placeholder intensity which identifies it as masked.
    """
    frameShape = sampleFrame.data.shape
    # generating the background data
    bgType = os.path.splitext(XDSPunchFilePath)[1]
    if bgType == ".h5":
        bg = xds_tools.bgFromH5(XDSPunchFilePath, frameShape)
    else:
        raise TypeError("Background file not supported!")

    helping_tools.check_folder(pathToPunched)
    frameset = cbf_tools.Frameset(pathToFrames, frameNameTemplate)

    # punching, the sorting function is used for a nice print out
    for fileName in frameset.generate_frame_names_from_template():
        print("Punching " + str(fileName), end="\r")
        frame = fabio.open(fileName)
        punched = frame.data - bg
        for x in range(frameShape[0]):
            for y in range(frameShape[1]):
                if punched[x,y] >= 0:
                    frame.data[x,y] = MASKED_BRAGG_INTENSITY
        frame.write(os.path.join(pathToPunched, punchedPrefix + os.path.basename(fileName)))
        del frame # freeing memory
    print("\nPunching complete!")


def update_masked_intensity(sampleFrame, pathToFrames, frameNameTemplate, pathToUpdated, updatedPrefix, updateIntensity=-1):
    """
    Changes the already provided masked intisities from MASKED_BRAGG_INTENSITY with to a new intensity.
    sampleFrame ... fabio.frame some frame from which the dimensions can be read.
    pathToFrames ... string location of the folder which contains the frames.
    frameNameTemplate ... string example of how the frames are named, allows for percent substitution.
    pathToPunched ... string location where the processed frames should be saved.
    punchedPrefix ... string prefix that should be put in front of the processed frames.
    """
    frameShape = sampleFrame.data.shape

    helping_tools.check_folder(pathToUpdated)
    frameset = cbf_tools.Frameset(pathToFrames, frameNameTemplate)

    # punching, the sorting function is used for a nice print out
    for fileName in frameset.generate_frame_names_from_template():
        print("Updating " + str(fileName), end="\r")
        frame = fabio.open(fileName)
        data = np.array(frame.data)
        data[data <= MASKED_BRAGG_INTENSITY] = updateIntensity
        frame.data = data
        frame.write(os.path.join(pathToUpdated, updatedPrefix + os.path.basename(fileName)))
        del frame # freeing memory
    print("\nUpdating complete!")


@helping_tools.deprecated
# is now in the mask and not reliable anyways
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