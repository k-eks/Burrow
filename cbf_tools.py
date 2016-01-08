import fabio
import os
import os.path
import glob
import shutil
import numpy as np


def get_flux(frame):
    """Reads the flux out of a frames header.
    frame ... fabio.frame the fabio class of the frame from which the flux needs to be known
    returns int the flux rate
    """
    flux = -1
    # header is a dict object, the flux is located under the given key
    if "Flux" in header:
        # get the flux by substring filtering
        flux = header[header.index("Flux") + 5 : header.index(" counts", header.index("Flux"))]
    else:
        raise AttributeError("Flux not found in frame %s!" % frame.filename)

    return int(flux)


def average_flux(pathToFrames):
    count = 0
    flux = 0
    for file in glob.glob(pathToFrames + "/*.cbf"):
        frame = fabio.open(file)
        flux += get_flux(fabio)
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


def sumup_frames(pathToFrames, sampleFrame, outputPath):
    """Makes a pixelwise summation of the frame data.
    pathToFrames ... string folder where the frames are located
    sampleFrame ... fabio.frame template for the data to read out the shape and to create a framework for the result
    outputPath ... string path to folder where the resulting frame should be saved
    """
    sampleFrame.data = np.zeros(sampleFrame.data.shape) # clear data
    for file in glob.glob(pathToFrames + "/*.cbf"):
        frame = fabio.open(file)
        print("Working on %s" % frame.filename, end='\r')
        sampleFrame.data += frame.data
        del frame # clean up memory
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
    for path, folders, files in os.walk(absolutePathToStart): # os.walk function requires the other variables
        for name in folders:
            fullPath = path + "/" + name + "/"
            if os.path.isfile(fullPath + "/" + conditionalFileName):
                print("Current folder is %s \n" % fullPath)
                print("Found conditional file, starting sumup")
                sumup_frames(fullPath, sampleFrame, outputPath)
            else:
                print("Skipping work in %s \n" % fullPath)


def get_set_revolution_size(pathToFrames):
    """Uses the common file name convention to read out the number of revolutions and the amount of frames per set from the file names in a given directory
    pathToFrames ... string path to where the cbf files are located
    returns setSize ... int the number of frames per revolution
    returns revolutions ... int the number revolutions
    """
    setSize = 0
    revolutionSize = 0
    for fileName in os.listdir(pathToFrames):
        # calculating the set size from frame name
        index = fileName.find("p_") # this is between the revolutions and frame number
        currentSize = int (fileName[index - 1 : index])
        revolutionSize = max(revolutionSize, currentSize)
        # calculating the revolution size from frame name
        index = fileName.find(".cbf")
        currentSize = int (fileName[index - 4 : index])
        setSize = max(setSize, currentSize)
    return setSize + 1, revolutionSize



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
                shutil.copy(pathToFrames + (nameTempalte % (r + 1, s - 1)), frameName)
                print("Inserted: %s\n" % frameName)
    print("\nDone!")


def folder_walker_lost_frame(absolutePathToStart, conditionalFileName, nameTempalte):
    """Performs a find_lost_frames_and_replace in a specific folder and all subfolders.
    absolutePathToStart ... string the absolute path to the starting direcotry which contains all sub direcotries
    conditionalFileName ... string only execute the find and replace this file is in folder, i.e. the first frame
    nameTempalte ... string template string from which the frame names are generated
    """
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