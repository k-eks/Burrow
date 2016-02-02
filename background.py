from __future__ import print_function # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import os.path
import fabio
import xds_tools
import h5py
import glob
import math

# some properties
CHUNK_SEQ = "sequential frame chunking"
CHUNK_SKI = "skipping frame chunking"
CHUNK_RAN = "random frame chunking"


def get_sequential_chunk(pathToFrames, nameTemplate, chunkStart, chunkSize):
    """Reads a defined number of frames in a squential manner into an array.
    pathToFrames ... string location of the folder which contains the frames
    nameTemplate ... string format of the frame names
    chunkStart ... int frame number from which to start
    chunkSize ... int number of frames which should be read
    returns numpy.array3d of all the frame data
    """
    # creating the file names for iteration
    fileNames = []
    for i in range(chunkSize):
        fileNames.append(pathToFrames + (nameTemplate % (i + chunkStart)))

    # generating the base data
    framePrototype = fabio.open(fileNames[0])
    stack = np.zeros((framePrototype.data.shape[0], framePrototype.data.shape[1], chunkSize), dtype=np.int32)

    # start stacking
    for i in range(chunkSize):
        print("Stacking, using " + str(fileNames[i]), end="\r")
        frame = fabio.open(fileNames[i])
        stack[:,:,i] = frame.data.copy()
        del frame # freeing memory
    print("\nStacking complete!")
    return stack


def get_skiping_chunk(pathToFrames, nameTemplate, frameRange, chunkStart, chunkSize):
    """Reads only every other frame and stacks their data to a chunk.
    pathToFrames ... string location of the folder which contains the frames
    nameTemplate ... string format of the frame names
    frameRange ... int maximum number of frames over which to run the algorithm
    chunkStart ... int frame number from which to start
    chunkSize ... int number of frames which should be read
    """
    # creating the file names for iteration
    fileNames = []
    for i in range(chunkSize):
        fileNames.append(pathToFrames + (nameTemplate % int(i * frameRange / chunkSize + chunkStart)))

    # generating the base data
    framePrototype = fabio.open(fileNames[0])
    stack = np.zeros((framePrototype.data.shape[0], framePrototype.data.shape[1], chunkSize), dtype=np.int32)

    # start stacking
    for i in range(chunkSize):
        print("Stacking, using " + str(fileNames[i]), end="\r")
        frame = fabio.open(fileNames[i])
        stack[:,:,i] = frame.data.copy()
        del frame # freeing memory
    print("\nStacking complete!")
    return stack


def get_percentile(chunk, frameShape, percentile):
    """Gets a percentile in z direction of a given data chunk."""
    bg = np.zeros((frameShape[0], frameShape[1]), dtype=np.int32)
    for x in range(frameShape[0]):
        for y in range(frameShape[1]):
            intensity = chunk[x,y,:]
            p = np.percentile(intensity, percentile)
            bg[x,y] = p
            print(" at %i %i" % (x, y), end="\r")
    print("\nPercentile calculation complete!")
    return bg


def generate_chunked_background_percentile(pathToFrames, pathToSubtracted, nameTemplate,
                                           frameRange, templateFrame, chunkSize,
                                           percentile, chunking=CHUNK_SKI):
    """Chunks frames and creates  partial background frames for these chunks.
    pathToFrames ... string location of the folder which contains the frames
    nameTemplate ... string format of the frame names
    frameRange ... int maximum number of frames over which to run the algorithm
    """
    for i in range(int(frameRange / chunkSize)):
        print("Using chunk " + str(i + 1) + " of " + str(int(frameRange / chunkSize)))
        # determination of the chunking method
        if chunking == CHUNK_SKI:
            chunk = get_skiping_chunk(pathToFrames, nameTemplate, frameRange, i + 1, chunkSize)
        elif chunking == CHUNK_SEQ:
            chunk = get_sequential_chunk(pathToFrames, nameTemplate, i * chunkSize + 1, chunkSize)
        bg = templateFrame # just as a prototype
        bg.data = get_percentile(chunk, templateFrame.data.shape, percentile)
        templateFrame.write(pathToSubtracted + "bg" + str(i) + ".cbf")
        del chunk, bg # cleaning memory
    print("\nFinished processing of all chunks with the method of \"%s\"!" % chunking)


def generate_subframe_background_percentile(pathToFrames, pathToBackground, nameTemplate,
                                            frameRange, subsize, percentile):
    """Creates a background by only reading in parts of frames and puzzeling these parts together.
    pathToFrames ... string location of the folder which contains the frames
    nameTemplate ... string format of the frame names
    """
    fileNames = []
    for i in range(frameRange):
        fileNames.append(pathToFrames + (nameTemplate % (i + 1)))

    templateFrame = fabio.open(fileNames[0]) # just a prototype
    bg = np.zeros((templateFrame.data.shape[0], templateFrame.data.shape[1]), dtype=np.int32)

    # determination of how many tiles are necessary for the subdivition of the frames
    tilesx = int(templateFrame.data.shape[0] / subsize) + 1
    tilesy = int(templateFrame.data.shape[1] / subsize) + 1
    for subx in range(tilesx):
        for suby in range(tilesy):
            print("\nWorking on sub %i of %i" % ((subx + 1) * (suby + 1), tilesx * tilesy))
            # generation of the subframe size taking the border regions into account
            if (subx + 2) > tilesx:
                width = templateFrame.data.shape[0] - subx * subsize
            else:
                width = subsize
            if (suby + 2) > tilesy:
                height = templateFrame.data.shape[1] - suby * subsize
            else:
                height = subsize
            print("Width %i, height %i" % (width, height))
            subFrame = np.zeros((width, height, frameRange))
            for i in range(frameRange):
                print("Reading frame " + fileNames[i], end="\r")
                frame = fabio.open(fileNames[i])
                subFrame[:, : , i] = frame.data[subx * subsize : subx * subsize + width,
                                                suby * subsize : suby * subsize + height].copy()
                del frame # cleaning memory
            print("\nCalculating percentile")
            bg[subx * subsize : subx * subsize + width,
               suby * subsize : suby * subsize + height] = get_percentile(subFrame, subFrame.shape, percentile)

    templateFrame.data = bg
    templateFrame.write(pathToBackground + "bg_subframe.cbf")


def generate_bg_chunked_master(pathToBgFrames, templateFrame, frameRange, percentile, mean=False):
    """Turns all generated background chunks into a single background frame.
    pathToBgFrames ... string location of the folder which contains the partial frame backgrounds
    frameRange ... int maximum number of frames over which to run the algorithm
    """
    bg = np.zeros((templateFrame.data.shape[0], templateFrame.data.shape[1]))
    chunk = get_sequential_chunk(pathToBgFrames, "bg%i.cbf", 0, frameRange)
    for i in range(frameRange):
        frame = fabio.open((pathToBgFrames + "bg%i.cbf" % i))
        bg = bg + frame.data
        del frame

    # create the finished background
    print("\n") # a precaution for a smooth output
    if percentile > 0: # gives the possibility to turn it off
        templateFrame.data = get_percentile(chunk, templateFrame.data.shape, percentile).astype(np.int32)
        templateFrame.write(pathToBgFrames + "bg_master_percentile" + str(percentile) + ".cbf")
        print("Created percentile background!")
    if mean: # no pun intended
        templateFrame.data = np.around(bg / frameRange).astype(np.int32)
        templateFrame.write(pathToBgFrames + "bg_master_mean.cbf")
        print("\nCreated mean background!")
        # usnig a different rounding function
        templateFrame.data = np.ceil(bg / frameRange).astype(np.int32)
        templateFrame.write(pathToBgFrames + "bg_master_ceil.cbf")
        print("\nCreated ceiled background!")


def subtract_background(bgFrame, hotFrame, pathToFrames, pathToSubtracted, hotPixelStart,
                        maskingOption, pathToBKGPIX):
    """Subtracts the background.
    pathToFrames ... string location of the folder which contains the frames
    """
    frameShape = bgFrame.data.shape
    bg = bgFrame.data
    print("Generating pixel mask.")

    # creating the mask
    if maskingOption == xds_tools.SIMPLEMASK:
        mask = xds_tools.read_all_unwanted_pixel(pathToBKGPIX, bgFrame)
    elif maskingOption == xds_tools.FULLMASK:
        defective, untrusted, hot = xds_tools.generate_all_unwanted_pixel(hotFrame, hotPixelStart)
    else:
        raise NotImplementedError("Other masking options are not yet implemented.")
    print("Mask generation done!")

    for imageFile in sorted(glob.glob(pathToFrames + "*.cbf"), key=xds_tools.numericalSort):
        print("Subtraction, using " + str(imageFile), end="\r")
        frame = fabio.open(imageFile)
        frame.data = (frame.data - bg).astype(np.int32)
        # restoring the mask
        if maskingOption == xds_tools.SIMPLEMASK:
            frame.data = xds_tools.restore_simple_pixel_mask(frame, mask)
        elif maskingOption == xds_tools.FULLMASK:
            frame.data = xds_tools.restore_pixel_mask(frame, defective, untrusted, hot)
        else:
            raise NotImplementedError("Other masking options are not yet implemented.")
        frame.write(pathToSubtracted + os.path.basename(imageFile))
        del frame # freeing some memory, otherwise memory would reach tera bytes at this point
    print("\nBackground subtraction complete!")