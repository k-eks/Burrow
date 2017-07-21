from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import scipy.signal
import os.path
import fabio
import xds_tools
import cbf_tools
import helping_tools
import h5py
import glob
import math


# some properties
CHUNK_SEQ = "sequential frame chunking"
CHUNK_SKI = "skipping frame chunking"
CHUNK_RAN = "random frame chunking"

@helping_tools.deprecated
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
    stack = np.zeros((framePrototype.data.shape[0], framePrototype.data.shape[1], chunkSize))

    # start stacking
    for i in range(chunkSize):
        print("Stacking, using " + str(fileNames[i]), end="\r")
        frame = fabio.open(fileNames[i])
        stack[:,:,i] = frame.data.copy()
        del frame # freeing memory
    print("\nStacking complete!")
    return stack


@helping_tools.deprecated
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
    stack = np.zeros((framePrototype.data.shape[0], framePrototype.data.shape[1], chunkSize))

    # start stacking
    for i in range(chunkSize):
        print("Stacking, using " + str(fileNames[i]), end="\r")
        frame = fabio.open(fileNames[i])
        stack[:,:,i] = frame.data.copy()
        del frame # freeing memory
    print("\nStacking complete!")
    return stack


def get_percentile(chunk, frameShape, percentile):
    """Gets a percentile in z direction of a given data chunk.
    chunk ... numpy.array3d(int) piece of data which contains the frame data of several frames, stacked in the third array dimension
    frameShape ... tuple(int, int) x and y dimesions of the frame
    percentile ... numeric percentile which should be calculated, range is from 0 to 100
    """
    bg = np.zeros((frameShape[0], frameShape[1]))
    for x in range(frameShape[0]):
        for y in range(frameShape[1]):
            intensity = chunk[x,y,:]
            p = np.percentile(intensity, percentile)
            bg[x,y] = p
            print(" at %i %i" % (x, y), end="\r")
    print("\nPercentile calculation complete!")
    return bg


@helping_tools.deprecated
def generate_chunked_background_percentile(pathToFrames, pathToJunks, nameTemplate,
                                           frameRange, templateFrame, chunkSize,
                                           percentile, chunking=CHUNK_SKI):
    """Chunks frames and creates  partial background frames for these chunks.
    pathToFrames ... string location of the folder which contains the frames
    pathToJunks ... string location where the individual junks should be stored
    nameTemplate ... string format of the frame names
    frameRange ... int maximum number of frames over which to run the algorithm
    templateFrame ... fabio.frame a frame from which the parameters of the measured frames can be deduced
    chunkSize ... int number of frames which should be read
    percentile ... numeric percentile which should be calculated, range is from 0 to 100
    chunking ... background.CONSTANT method which should be used to generate the chunks
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
        templateFrame.write(pathToJunks + "bg" + str(i) + ".cbf")
        del chunk, bg # cleaning memory
    print("\nFinished processing of all chunks with the method of \"%s\"!" % chunking)


def generate_subframe_background_percentile(pathToFrames, pathToBackground, nameTemplate,
                                            frameRange, subsize, percentile, outputName,
                                            outputModifiers=None):
    """Creates a background by only reading in parts of frames and puzzeling these parts together.
    pathToFrames ... string location of the folder which contains the frames
    pathToBackground ... string location where the background frame should be placed
    nameTemplate ... string format of the frame names, allows percent substitution
    frameRange ... int maximum number of frames over which to run the algorithm
    subsize ... int number of pixels in x and y directions to determine the subframe size
                this is used to save memory
    percentile ... numeric the percentile of the frames which should be considered as background
    outputName ... string name of the finished background frame, allows percent substituiton
    outputModifiers ... string plus-sign seperated string list, these modfieres are used to susbtitute outputName
    """
    # parse the modifiers
    outputModifiers = helping_tools.parse_substition_modifiers(outputModifiers)
    fileNames = []
    frameset = cbf_tools.Frameset(pathToFrames, nameTemplate)
    fileNames = frameset.generate_frame_names_from_template()

    templateFrame = fabio.open(fileNames[0]) # just a prototype
    bg = np.zeros((templateFrame.data.shape[0], templateFrame.data.shape[1]))

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

    helping_tools.check_folder(pathToBackground)
    # create and write the flux monitor
    fluxFileName = "fluxmonitor_" + outputName + ".csv"
    flux = cbf_tools.average_flux(pathToFrames, nameTemplate, pathToBackground, fluxFileName % outputModifiers, frameRange)
    # writing the background file
    templateFrame.data = bg.astype(np.int32)
    fileName, fileExtension = os.path.splitext(outputName)
    # splicing the average flux into the file name and prepare the extension
    outputName = fileName + "_flux_" + str(flux) + ".cbf"
    # write the cbf file
    templateFrame.write(os.path.join(pathToBackground, outputName % outputModifiers))
    # write the h5 file
    cbf_tools.frame_to_h5(templateFrame, os.path.join(pathToBackground, outputName + ".h5"), outputModifiers)


@helping_tools.deprecated
def generate_bg_chunked_master(pathToBgFrames, templateFrame, frameRange, percentile, mean=False):
    """Turns all generated background chunks into a single background frame.
    pathToBgFrames ... string location of the folder which contains the partial frame backgrounds
    templateFrame ... fabio.frame a frame from which the parameters of the measured frames can be deduced
    frameRange ... int maximum number of frames over which to run the algorithm
    percentile ... numeric percentile which should be calculated, range is from 0 to 100
    mean ... bool generates two additional background frames based on the mean and median
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


def subtract_hybrid_background(pathToFrames, pathToSubtracted, backgroundFrameNames,
                               backgroundMixture, bgName, maskFrame):
    """Generates a flux normalized background from multiple sources and subtracts it from the raw data.
    pathToFrames ... string location of the folder which contains the frames
    pathToSubtracted ... string location where the processed frames should be saved
    backgroundFrameNames ... array[string] names of the background files which should be used
    backgroundMixture ... array[float] contribution of each background frame for the final background image
    bgName ... string prefix which should be added to the modified frames
    maskFrame ... fabio.frame frame which contains all pixel that should be masked
    """
    bgFluxes = []
    bgData = []
    bgCount = len(backgroundFrameNames)
    for fileName in backgroundFrameNames:
        bgFluxes.append(get_flux_from_file_name(fileName))
        # grab the file extension, NOTE: extension includes the dot!!!!!!!!!!!
        fileType = os.path.splitext(fileName)[1]
        if fileType == '.h5':
            bgData.append(cbf_tools.h5_to_numpy(fileName, ()))
        elif fileType == '.cbf':
            bgData.append(fabio.open(fileName).data)

    helping_tools.check_folder(pathToSubtracted)
    print("Reading masks, please wait!")
    maskUntrusted, maskDefective, maskHot = cbf_tools.generate_all_unwanted_pixel(maskFrame, 1000000)
    print("starting subtracting\n")
    frameset = cbf_tools.Frameset(pathToFrames, backgroundFrameNames)
    for fileName in frameset.generate_frame_names_from_template():
        frame = fabio.open(fileName)
        frameFlux = cbf_tools.get_flux(frame)
        # mix the background frame
        bgAll = np.zeros(frame.data.shape)
        for i in range(bgCount):
            scale = frameFlux / bgFluxes[i]
            bgAll += bgData[i] * scale * backgroundMixture[i]
        frame.data -= bgAll # here is the actual backround subtraction
        frame.data = frame.data.round().astype(np.int32) # make resonable counts
        frame.data = cbf_tools.restore_pixel_mask(frame, maskUntrusted, maskDefective, maskHot)
        fileName = os.path.basename(fileName) # preparing writing to new location
        frame.save(os.path.join(pathToSubtracted, bgName + fileName))
        print("Background subtracted from %s" % fileName, end='\r')
        del frame # cleaning up memory
    print("\nDone!")


def subtract_single_frame(pathToFrames, pathToSubtracted, namePrefix, single, maskFrame):
    """Subtracts a flux normalized frame from a dataset.
    pathToFrames ... string location of the folder which contains the frames
    pathToSubtracted ... string location where the processed frames should be saved
    namePrefix ... string short text that is added to each newly calculated frame
    single ... fabio.frame this frame will be substracted from the dataset
    maskFrame ... fabio.frame frame which contains all pixel that should be masked
    """
    helping_tools.check_folder(pathToSubtracted)
    print("Reading subtraction frame...")
    singleFlux = cbf_tools.get_flux(single)
    print("Reading masks, please wait!")
    maskUntrusted, maskDefective, maskHot = cbf_tools.generate_all_unwanted_pixel(maskFrame, 1000000)
    print("starting subtracting\n")
    frameset = cbf_tools.Frameset(pathToFrames)
    for fileName in frameset.generate_frame_names_from_template():
        singleData = single.data.copy()
        frame = fabio.open(fileName)
        frameFlux = cbf_tools.get_flux(frame)
        # normalizing intensity onto the flux
        fluxRatio = frameFlux / singleFlux
        singleData = singleData * fluxRatio
        frame.data -= singleData.astype(np.int32) # here is the actual frame subtraction
        frame.data = frame.data.round().astype(np.int32) # make resonable counts
        frame.data = cbf_tools.restore_pixel_mask(frame, maskUntrusted, maskDefective, maskHot)
        fileName = os.path.basename(fileName) # preparing writing to new location
        frame.save(os.path.join(pathToSubtracted, namePrefix + fileName))
        print("Frame subtracted from %s" % fileName, end='\r')
        del frame # cleaning up memory
        del singleData # cleaning up memory
    print("\nDone!")


@helping_tools.deprecated
# this function is incomplete, the SavGol filter is not reliable around Bragg peaks ****************************************************
def SavGol_filter(pathToFrames, nameTemplate, frameRange, pathToFiltered, namePrefix, maskFrame, subsize, windowLength, polyOrder):
    """Subtracts a flux normalized frame from a dataset.
    pathToFrames ... string location of the folder which contains the frames
    pathToSubtracted ... string location where the processed frames should be saved
    namePrefix ... string short text that is added to each newly calculated frame
    single ... fabio.frame this frame will be substracted from the dataset
    maskFrame ... fabio.frame frame which contains all pixel that should be masked
    """
    helping_tools.check_folder(pathToFiltered)
    print("Reading masks, please wait!")
    # maskUntrusted, maskDefective, maskHot = cbf_tools.generate_all_unwanted_pixel(maskFrame, 1000000)
    print("starting filtering\n")
    # generating frame paths and names for reading
    frameset = cbf_tools.Frameset(pathToFrames, nameTemplate)
    frameset.setSize = frameRange
    fileNames = frameset.generate_frame_names_from_template()
    # generating frame paths and names for writing
    frameset = cbf_tools.Frameset(pathToFiltered, namePrefix + nameTemplate)
    frameset.setSize = frameRange
    newFiles = frameset.generate_frame_names_from_template()

    templateFrame = fabio.open(fileNames[0])
    # determination of how many tiles are necessary for the subdivition of the frames
    tilesx = int(templateFrame.data.shape[0] / subsize) + 1
    tilesy = int(templateFrame.data.shape[1] / subsize) + 1
    for subx in range(tilesx):
        for suby in range(tilesy):
            print("\nWorking on sub %i of %i" % ((suby * (subx + 1) + subx), tilesx * tilesy))
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
                print("Reading frame " + fileNames[i])#, end="\r")
                frame = fabio.open(fileNames[i])
                subFrame[:, : , i] = frame.data[subx * subsize : subx * subsize + width,
                                                suby * subsize : suby * subsize + height].copy()
                del frame # cleaning memory
            print("\nApplying SavGol filter...")
            for x in range(subFrame.shape[0]):
                for y in range(subFrame.shape[1]):
                    print(x, y, end="\r")
                    filterLine = subFrame[x, y, :]
                    subFrame[x, y, :] = scipy.signal.savgol_filter(filterLine, windowLength, polyOrder, mode='wrap').copy()
            subframe = subFrame.astype(np.int32)
            for i in range(frameRange):
                print("Writing frame " + newFiles[i])#, end="\r")
                frame = fabio.open(newFiles[i])
                frame.data[subx * subsize : subx * subsize + width, suby * subsize : suby * subsize + height] = subFrame[:, : , i]
                frame.save(os.path.join(pathToFiltered, namePrefix + os.path.basename(frame.filename)))
                del frame # cleaning memory
    print("\nDone!")


# this function is inaptly named
@helping_tools.deprecated
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


def get_flux_from_file_name(fileName):
    """Reads out the flux written in the file path of the h5 background files.
    fileName ... string name of the the file from which the flux should be extracted, path optional
    returns int flux of the file
    """
    fileName = os.path.basename(fileName)
    # flux is encoded between the last underscore and the first dot
    flux = int(fileName.split('_')[-1].split('.')[0])
    print("Flux of %s is %s" % (fileName, flux))
    return flux