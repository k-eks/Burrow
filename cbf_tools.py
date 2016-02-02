from __future__ import print_function # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")


import fabio
import os
import os.path
import glob
import shutil
import numpy as np
import helping_tools

DEFAULT_FRAME_NAME = "frame_0_0_80_%04ip_%05i.cbf"


def get_flux(frame):
    """Reads the flux out of a frames header.
    frame ... fabio.frame the fabio class of the frame from which the flux needs to be known
    returns int the flux rate
    """
    flux = -1
    headerData = frame.header['_array_data.header_contents']
    # header is a dict object, the flux is located under the given key
    if "Flux" in headerData:
        # get the flux by substring filtering
        flux = headerData[headerData.index("Flux") + 5 : headerData.index(" counts", headerData.index("Flux"))]
    else:
        # this would be a very serious problem
        raise AttributeError("Flux not found in frame %s!" % frame.filename)

    return int(flux)


def average_flux(pathToFrames):
    count = 0
    flux = 0
    for file in glob.glob(pathToFrames + "/*.cbf"):
        frame = fabio.open(file)
        flux += get_flux(frame)
        del frame
        count += 1 # counts the number of frames
    return int(flux / count) # averaging


def create_bg_frame(pathToFrames, sampleFrame):
    count = 0
    flux = 0
    for file in glob.glob(pathToFrames + "/*.cbf"):
        frame = fabio.open(file)


def get_main_folder_name(path):
    """Gets the name of the last folder in path but takes one folder up higher if the last folder is called 'frames'.
    path ... string the path from which the last direcotry is to be extracted
    returns string the name of the last directory
    """
    parts = path.split('/')
    # remove empty entries caused by leading and trailing slashes
    while '' in parts:
        parts.remove('')
    l = len(parts)
    # if the last directory in the tree is called 'frames', use the second last
    if parts[l - 1] == "frames":
        modifier = 1
    else:
        modifier = 0
    return parts[l - 1 - modifier]


def find_faulty_frames(pathToFrames):
    """Checks if all frames are readable by fabio.
    pathToFrames ... string path to the folder which contains the frames
    returns ... list[string] all files which are faulty
    """
    print("Searching for faulty frames in %s" % pathToFrames)
    faulty = []
    for file in glob.glob(pathToFrames + "/*.cbf"):
        try:
            frame = fabio.open(file)
            print("Working on %s" % frame.filename, end='\r')
            del frame # clean up memory
        except Exception:
            faulty.append(file) # create a list of file names
    print("\nDone, found %s faulty frames in %s.\n%s" % (len(faulty), pathToFrames, faulty))
    return faulty


def sumup_frames(pathToFrames, sampleFrame, outputPath):
    """Makes a pixelwise summation of the frame data.
    pathToFrames ... string folder where the frames are located
    sampleFrame ... fabio.frame template for the data to read out the shape and to create a framework for the result
    outputPath ... string path to folder where the resulting frame should be saved
    """
    sampleFrame.data = np.zeros(sampleFrame.data.shape) # clear data
    for file in glob.glob(pathToFrames + "/*.cbf"):
        try:
            frame = fabio.open(file)
            print("Working on %s" % frame.filename, end='\r')
            sampleFrame.data += frame.data
            del frame # clean up memory
        except PermissionError:
            print("\nNot allowed to read: %s\n" % file)
    outputPath = outputPath + "/sumup_" + get_main_folder_name(pathToFrames) + ".cbf"
    print("\nWriting to %s" % outputPath)
    sampleFrame.save(outputPath) # this does not overwrite the original frame used as a sample
    # somehow the header gets corrupted by the previous statement
    print("Writing header, please wait!")
    frame = fabio.open(outputPath) # re-reading and writing the same file again fixes the header (only the gods know why...)
    frame.save(outputPath)
    print("Done!")


def folder_walker_sumup(absolutePathToStart, conditionalFileName, sampleFrame, outputPath):
    """Performs a sumup_frames in a specific folder and all subfolders.
    absolutePathToStart ... string the absolute path to the starting direcotry which contains all sub direcotries
    conditionalFileName ... string only execute the find and replace this file is in folder, i.e. the first frame
    sampleFrame ... fabio.frame template for the data to read out the shape and to create a framework for the result
    outputPath ... string path to folder where the resulting frame should be saved
    """
    # TODO: reduce to frame folders with helping_tools.find_named_folders?
    for path, folders, files in os.walk(absolutePathToStart): # os.walk function requires the other variables
        for name in folders:
            fullPath = path + "/" + name + "/"
            if os.path.isfile(fullPath + "/" + conditionalFileName):
                print("Current folder is %s \n" % fullPath)
                print("Found conditional file, starting sumup")
                sumup_frames(fullPath, sampleFrame, outputPath)
            else:
                print("Skipping work in %s \n" % fullPath)


def get_set_revolution_size_dataset(pathToFrames):
    """Uses the common file name convention to read out the number of revolutions and the amount of frames per set from the file names in a given directory.
    pathToFrames ... string path to where the cbf files are located
    returns setSize ... int the number of frames per revolution
    returns revolutions ... int the number revolutions
    """
    setSize = 0
    revolutionSize = 0
    for fileName in os.listdir(pathToFrames):
        if ".cbf" in fileName:
            # calculating the set size from frame name
            index = fileName.find("p_") # this is between the revolutions and frame number
            currentSize = int (fileName[index - 1 : index])
            revolutionSize = max(revolutionSize, currentSize)
            # calculating the revolution size from frame name
            index = fileName.find(".cbf")
            currentSize = int (fileName[index - 4 : index])
            setSize = max(setSize, currentSize)
    return setSize + 1, revolutionSize


def get_set_revolution_size_frame(fileName):
    """Uses the common file name convention to read out the current revolution and the frame number
    fileName ... string path to where the frame is located
    returns frameNumber ... int the number of frames per revolution
    returns revolutions ... int the number revolutions
    """
    # calculating the set size from frame name
    index = fileName.find("p_") # this is between the revolutions and frame number
    revolution = int (fileName[index - 1 : index])
    # calculating the revolution size from frame name
    index = fileName.find(".cbf")
    frameNumber = int (fileName[index - 4 : index])
    return frameNumber, revolution


def rename_frames(pathToFrames, newName):
    """Changes the part before the underscore in the frame filename.
    pathToFrames ... string location of the frames
    newName ... string the new idetifier for the frame
    """
    print("starting renaming\n")
    for fileName in glob.glob(os.path.join(pathToFrames, "*.cbf")):
        # make sure that no underscores in folders are taken as a starting point
        fileName = os.path.basename(fileName)
        length = fileName.find("_")
        newFileName = newName + fileName[length + 1 :]
        os.rename(os.path.join(pathToFrames, fileName), os.path.join(pathToFrames, newFileName))
        print("Renamed %s" % newFileName, end='\r')
    print("\nDone!")


def find_lost_frames_and_replace(pathToFrames, nameTempalte, setSize, revolutions):
    """Looks for inconsequential numbering in the cbf files and replaces missing frame numbers with copies of the previous frame.
    pathToFrames ... string location of the frames
    nameTempalte ... string  a two times percent subsituted string of the file name which contains revolutions and frame number
    setSize ... int number of frames per revolution which should be present
    revolutions ... int number of revolutions
    """
    for r in range(revolutions):
        for s in range(setSize):
            frameName = pathToFrames + (nameTempalte % (r + 1, s))
            print("Working on %s" % frameName, end='\r')
            if not os.path.isfile(frameName):
                print("\nMissing: ", frameName)
                # replace the missing frame with the previous one
                previousFrameName = pathToFrames + (nameTempalte % (r + 1, s - 1))
                shutil.copy(previousFrameName, frameName)
                os.chmod(frameName, 0o644) # if working on a server, the rights might get screwed up
                # 0o is an octal number, rigths are set according to UNIX
                print("Inserted: %s\n" % frameName)
    print("\nDone!")


def folder_walker_lost_frame(absolutePathToStart, conditionalFileName, nameTempalte):
    """Performs a find_lost_frames_and_replace in a specific folder and all subfolders.
    absolutePathToStart ... string the absolute path to the starting direcotry which contains all sub direcotries
    conditionalFileName ... string only execute the find and replace this file is in folder, i.e. the first frame
    nameTempalte ... string template string from which the frame names are generated
    """
    # TODO: reduce to frame folders with helping_tools.find_named_folders?
    for path, folders, files in os.walk(absolutePathToStart): # os.walk function requires the other variables
        for name in folders:
            fullPath = path + "/" + name + "/"
            if os.path.isfile(fullPath + "/" + conditionalFileName):
                print("Current folder is %s \n" % fullPath)
                print("Found conditional file, calculating set and revolution size...")
                setSize, revolutions = get_set_revolution_size(fullPath)
                print("Set size is %s, revolutions is %s" % (setSize, revolutions))
                find_lost_frames_and_replace(fullPath, nameTempalte, setSize, revolutions)
            else:
                print("Skipping work in %s \n" % fullPath)


def find_and_rename_zero_frame(pathToFrames, searchSequence, replaceSequence):
    """Looks for frames which have a given sequence in their file name and replaces it.
    pathToFrames ... string path where to look for the frames
    searchSequence ... string all files which contain this string will be renamed
    replaceSequence ... string this string replaces the the found string in the file name
    """
    searchString = "*%s*" % searchSequence
    searchString = os.path.join(pathToFrames, searchString)
    for fileName in glob.glob(searchString):
        newFileName = fileName.replace(searchSequence, replaceSequence)
        os.rename(fileName, newFileName)
        print("Renamed %s in folder %s" % (fileName, pathToFrames))


def folder_walker_find_and_rename_zero_frame(pathToRoot, searchSequence, replaceSequence):
    """Performs a find_and_rename_zero_frame in all frame folders below a given start point.
    pathToRoot ... string path where to look walk down and look for the frame folders
    searchSequence ... string all files which contain this string will be renamed
    replaceSequence ... string this string replaces the the found string in the file name
    """
    folders = helping_tools.find_named_folders(pathToRoot, "frames")
    for folder in folders:
        print("Current folder is %s" % folder)
        find_and_rename_zero_frame(folder, searchSequence, replaceSequence)
    print("Done!")


def split_revolutions(pathToFrames, nameTempalte):
    """Creates subfolders for each revolution and moves the corresponding frames.
    pathToFrames ... string path where to look for the frames
    nameTempalte ... string template string from which the frame names are generated
    """
    setSize, revolutions = get_set_revolution_size_dataset(pathToFrames)
    print("Starting moving...")
    for i in range(revolutions):
        subPath = os.path.join(pathToFrames, "rev%s" %(i + 1))
        os.makedirs(subPath)
        print("\nCreated %s" % subPath)
        print("Moving revolution %s of %s" % (i + 1, revolutions))
        for j in range(setSize - 1):
            # setup copy path and change to 1-based counting
            source = os.path.join(pathToFrames, nameTempalte % (i + 1, j + 1))
            destination = os.path.join(subPath, nameTempalte % (i + 1, j + 1))
            shutil.move(source, destination)
            print("Moved %s" % os.path.basename(source), end='\r')