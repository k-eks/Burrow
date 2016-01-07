import fabio
import os
import os.path
import shutil


def get_Flux(frame):
    """Reads the flux out of a frames header.
    frame ... fabio.frame the fabio class of the frame from which the flux needs to be known
    returns int the flux rate
    """
    foundFlux = False
    flux = -1
    # header is a dict object, the flux is located under the given key
    if "Flux" in header:
        # get the flux by substring filtering
        flux = header[header.index("Flux") + 5 : header.index(" counts", header.index("Flux"))]
        foundFlux = True

    if not foundFlux:
        raise AttributeError("Flux not found!")

    return int(flux)


def get_set_revolution_size(pathToFrames):
    """Uses the common file name convention to read out the number of revolutions and the amount of frames per set from the file names in a given directory
    pathToFrames ... string path to where the cbf files are located
    returns setSize ... int the number of frames per revolution
            revolutions ... int the number revolutions
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


def lost_frame_folder_walker(absolutePathToStart, conditionalFileName, nameTempalte):
    """Performs a find lost frames and replace in a specific folder and all subfolders.
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