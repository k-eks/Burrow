from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import helping_tools
import math
import scipy
import h5py
import os
import os.path
import scipy.ndimage
import pixdata


# two lists with yell's standard file names for reciprocal and real space data sets
ReciprocaleSpaceFiles = ["experiment.h5", "model.h5", "exp-minus-model.h5"]
PdfSpaceFiles = ["exp-delta-pdf.h5", "delta-pdf.h5", "delta-delta-pdf.h5"]


def make_array_fft_ready(data,NAN_to_zero=True):
    """
    Cuts off the last indices of an odd array in order to prevent sepctral leakage during the FFT. Works with 2D and 3D arrays.
    data ... numpy.array The array which is to be pruned.
    NAN_to_zero ... bool wether or not NANs should be changed to zeros.
    returns ... numpy.array The modified array without the odd indices.
    """
    result = None
    if len(data.shape) == 2:
        if (data.shape[0] % 2 == 1) and (data.shape[1] % 2 == 1):   # only do something if all array indices are odd
            result = np.array(data[:-1, :-1]) # deep copy of the result
        else:
            raise IndexError("Please provide an array where all indices are odd!")
    elif len(data.shape) == 3:
        if (data.shape[0] % 2 == 1) and (data.shape[1] % 2 == 1) and (data.shape[2] % 2 == 1):   # only do something if all array indices are odd
            result = np.array(data[:-1, :-1, :-1]) # deep copy of the result
        else:
            raise IndexError("Please provide an array where all indices are odd!")
    else:
        raise IndexError("Input array has an unkown shape!")

    if NAN_to_zero:
        result = np.nan_to_num(result)
    return result


def flip_data_set(inputFileName):
    """
    Inverts a data set when XDS has found the wrong handiness.
    inputFileName ... string Specifies the file which is to be inverted.
    """
    inputFile = h5py.File(inputFileName, 'a')
    data = np.array(inputFile['data'])
    del inputFile['data']
    inputFile['data'] = np.flip(data,2)
    inputFile.close()


def fft_dataset(inputFileName, fileForKeys=None):
    """
    Calculates the absolute, real and imaginary part of the Fourier Transform for a given yell dataset file.
    inputFileName ... string Specifies the file which is to be transformed.
    fileForKeys ... string (optional) source file from which to copy the hdf keys to the newly created files.
    """
    inputFile = h5py.File(inputFileName, 'r')

    data = np.array(inputFile['data'])
    print("Shape is %s, make sure it is the correct shape for FFT!" % str(data.shape))
    print("Calculating FFT...")
    data = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(data)))

    outputFile = h5py.File("t_r_" + inputFileName, 'w')
    print("Writing real part...")
    outputFile['data'] = np.real(data)
    outputFile.close()
    del outputFile
    outputFile = h5py.File("t_i_" + inputFileName, 'w')
    print("Writing imaginary part...")
    outputFile['data'] = np.imag(data)
    outputFile.close()
    del outputFile
    outputFile = h5py.File("t_a_" + inputFileName, 'w')
    print("Writing absolute part...")
    outputFile['data'] = np.abs(data)
    outputFile.close()
    del outputFile

    if fileForKeys != None:
        transplant_keys(fileForKeys, "t_r_" + inputFileName)
        transplant_keys(fileForKeys, "t_i_" + inputFileName)
        transplant_keys(fileForKeys, "t_a_" + inputFileName)


def create_mask_from_nan(inputFileName, outputFileName):
    """
    Takes all values from an input file and transforms all NANs to zero and all non-NANs to 1.
    The result is saved in a new file and should be used as a mask.
    inputFileName ... string path to file which should be read
    outputFileName ... string file name to which the mask should be saved.
    """
    inputFile = h5py.File(inputFileName, 'r')
    maskFile = h5py.File(outputFileName, 'w')

    mask = np.array(inputFile['data'])
    mask[np.isnan(mask)] = 0
    mask[mask>0] = 1
    maskFile['data'] = mask
    # cleaning up and create keys
    maskFile.close()
    del maskFile
    transplant_keys(inputFileName, outputFileName)


def transplant_keys(keysFromPath, keysToPath, verbose=True):
    """
    Copies the metadata keys from one file to another. The dataset with the key 'data' is excluded.
    keysFromPath ... string Path to the file from which the metadata should be taken.
    keysToPath ... string Path to the file into which the metadata should be pasted.
    verbose ... bool Wheter or not more details on the copying process should be given.
    """
    keysFrom = h5py.File(keysFromPath, mode='r')
    keysTo = h5py.File(keysToPath, mode='a')

    keys = list(keysFrom.keys())
    keys.remove('data') # we don't want to overwrite the data, just the metadata
    for item in keys:
        if verbose:
            print("Copying key %s" % item)
        if item not in list(keysTo):
            keysTo[item] = np.array(keysFrom[item])
        elif verbose:
            print("Key %s already exists!" % item)
    # clean up
    keysFrom.close()
    keysTo.flush()
    keysTo.close()


def increase_contrast(inputFileName, factor=100):
    """
    Mulitplies an h5 yell-file with a factor to give a better contrast in the PDF viewer.
    inputFileName ... string path to file which should be modified
    factor ... float number with which all data is multiplied
    """
    file = h5py.File(inputFileName)
    file['data'][:,:,:] = file['data'][:,:,:] * factor
    file.close()


def get_wR(fobs,fcalc,weights):
    """
    Calculates the weighted R value of two numpy arrays.
    fobs ... numpy.array observed data
    fcalc ... numpy.array model data
    weights ... numpy.array weights (or mask) applied to data
    returns float the calculated wR value
    """
    return math.sqrt(np.sum(np.abs((weights * (np.abs(fobs - fcalc)) ** 2))) / np.sum(np.abs(weights * fobs ** 2)))


def create_PDF_mask(baseFile, outputFileName, size=40):
    """
    Creates a mask file in the shape of a hexagonal prism.
    baseFile ... string path to a h5 file to get the metadata for the prism
    outputFileName ... string name of the newly created file which contains the mask
    size ... int diameter of the mask in pixel
    """
    # clean up former files
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    mask = h5py.File(outputFileName)

    # creating helper variables to work with the shape
    dataShape = np.array(h5py.File(baseFile)['data'].shape)
    center = dataShape // 2

    # forming a broad outline in the shape of a prism
    data = np.zeros(dataShape)
    data[center[0] - size // 2:center[0] + size // 2, center[1] - size // 2:center[1] + size // 2, center[2] - size // 2:center[2] + size // 2] = 1

    # cutting of one edge of the original prism
    lowerright = data[center[0]:center[0] + size // 2,center[1] - size // 2:center[1], center[2] - size // 2:center[2] + size // 2]
    for i in range(lowerright.shape[0]):
        for j in range(lowerright.shape[1]):
            if i - j >  0:
                lowerright[i,j,:] = 0 # upper left
            else:
                lowerright[i,j,:] = 1 # llower right
    data[center[0]:center[0] + size // 2,center[1] - size // 2:center[1], center[2] - size // 2:center[2] + size // 2] = lowerright

    # cutting of the second edge of the original prism to create a hexagonal shape
    upperleft = data[center[0] - size // 2:center[0], center[1]:center[1] + size // 2, center[2] - size // 2:center[2] + size // 2]
    for i in range(upperleft.shape[0]):
        for j in range(upperleft.shape[1]):
            if i - j > 0:
                upperleft[i,j,:] = 1 # upper left
            else:
                upperleft[i,j,:] = 0 # lower right
    data[center[0] - size // 2:center[0], center[1]:center[1] + size // 2, center[2] - size // 2:center[2] + size // 2] = upperleft

    mask['data'] = data
    transplant_keys(baseFile, outputFileName)


def calculate_rw(pathToExp, pathToModel, pathToWeights=None):
    """
    Calculates the weigted Rw of two data sets.
    pathToExp ... string file path to the experimental data
    pathToModel ... string file path to the model data
    pathToWeights ... string (optional) file path to the weights
    """
    rPlain = get_wR(np.array(h5py.File(pathToExp)['data']), np.array(h5py.File(pathToModel)['data']), 1)
    print("Rw = ", rPlain)
    if pathToWeights != None:
        weights = np.array(h5py.File(pathToWeights)['data'])
        Rw = get_wR(np.array(h5py.File(pathToExp)['data']), np.array(h5py.File(pathToModel)['data']), weights)
        print("Masked Rw = ", Rw)



def splice_data_uvx(inputFiles, outputFileName, filePath="", bgFileName=None, bgMultiplier=[1, 1, 0]):
    """
    Uses three direct space data sets and creats a combined view.
    Perferable used for uvx data sets
    inputFiles ... array[string] array of THREE strings, each strings is a file path to one of the data sets which should be combined.
    outputFileName ... string name of the newly created file which combines the three input files.
    filePath ... str this path will be attached to the provided data sets
    """
    print("WARNING: USES HARDCODED ARRAY SIZE!")
    # applying file paths
    outputFileName = os.path.join(filePath, outputFileName)
    for i in range(len(inputFiles)):
        inputFiles[i] = os.path.join(filePath, inputFiles[i])

    # clean up former files
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    outputFile = h5py.File(outputFileName)
    # read out data and apply background
    if bgFileName != None:
        BG = np.array(h5py.File(os.path.join(filePath, bgFileName))['data'][:,:,:])
    else:
        BG = 0
    f1 = np.array(h5py.File(inputFiles[0])['data'][:,:,:]) - BG * bgMultiplier[0]
    f2 = np.array(h5py.File(inputFiles[1])['data'][:,:,:]) - BG * bgMultiplier[1]
    f3 = np.array(h5py.File(inputFiles[2])['data'][:,:,:]) - BG * bgMultiplier[2]

    data = np.array(f1) # for size
    # creating a mask for where to put the data
    data[105:,:,:] = -100 # upper right
    data[:,:105,:] = -200 # lower right
    data[:105,:,:] = -300 # left
    # the lower left quadrant belongs partially to left and partially to lower right
    # in the following, it will be distributed
    lowerleft = data[:105,:105,:]
    for i in range(lowerleft.shape[0]):
        for j in range(lowerleft.shape[1]):
            if i - j > 0:
                lowerleft[i,j,:] = -200 # lower right
            else:
                lowerleft[i,j,:] = -300 # left
    data[:105,:105,:] = lowerleft

    # applying the data to its destination
    data[data == -100] = f1[data == -100]
    data[data == -200] = f2[data == -200]
    data[data == -300] = f3[data == -300]
    transplant_keys(inputFiles[0], outputFileName)
    outputFile['data'] = data


def splice_data_hkx(inputFiles, outputFileName, filePath=""):
    """
    Uses three reciprocal space data sets and creats a combined view.
    Perferable used for hkx reconstructions but has limited support for 0kl as well.
    inputFiles ... array[string] array of THREE strings, each strings is a file path to one of the data sets which should be combined.
    outputFileName ... string name of the newly created file which combines the three input files.
    filePath ... str this path will be attached to the provided data sets
    """
    print("WARNING: USES HARDCODED ARRAY SIZE!")
    # applying file paths
    outputFileName = os.path.join(filePath, outputFileName)
    for i in range(len(inputFiles)):
        inputFiles[i] = os.path.join(filePath, inputFiles[i])

    # clean up former files
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    outputFile = h5py.File(outputFileName)
    # read out data
    f1 = h5py.File(inputFiles[0])
    f2 = h5py.File(inputFiles[1])
    f3 = h5py.File(inputFiles[2])
    data = np.array(f1['data'])
    # creating a mask for where to put the data
    data[105:,:,:] = -100 # upper right
    data[:,:105,:] = -200 # lower right
    data[:,:,:155] = -200 # filling up lower half
    data[:105,:,:] = -300 # left

    # the upper left quadrant belongs partially to left and partially to upper right
    # in the following, it will be distributed
    upperleft = data[:105,105:,:]
    for i in range(upperleft.shape[0]):
        for j in range(upperleft.shape[1]):
            if i + j < upperleft.shape[0]:
                upperleft[i,j,:] = -300 # lower right
            else:
                upperleft[i,j,:] = -100 # left
    data[:105,105:,:] = upperleft

    # applying the data to its
    data[data == -100] = f1['data'][data == -100]
    data[data == -200] = f2['data'][data == -200]
    data[data == -300] = f3['data'][data == -300]
    transplant_keys(inputFiles[0], outputFileName)
    outputFile['data'] = data


def subtract_meerkat_dataset(dataset, subtrahend, outputFileName=None):
    """
    This function subtracts 3D data in different formats from each other.
    A key feature of this function is its power to parse multiple data types into each other and back so that I don't have to worry what kind of data I feed it to.
    dataset ... 3D data from which subtrahend will be subtracted
    subtrahend ... 3D data which will be subtracted from dataset
    the input types are special, for dataset and subtrahend they can be either str, a h5 file, a h5 dataset or np.array and not neccessarily the same for both.
    outputFileName ... str when given and dataset is also a path, the result will be written to this file
    return np.array the result of the subtraction
    """
    print("WARNING: ONLY WORKS WITH THREE DIMENSIONAL DATA!!!")
    resultType = type(dataset)
    # the follwoing two variables will hold the values for subtraction
    data = None
    sub = None

    # parsing input
    # check if inputs are files paths
    if type(dataset) == str:
        data = np.array(h5py.File(dataset)['data'][:,:,:])
    if type(subtrahend) == str:
        sub = np.array(h5py.File(subtrahend)['data'][:,:,:])

    # check if inputs are h5-files
    if type(dataset) == h5py._hl.files.File:
        data = np.array(dataset['data'][:,:,:])
    if type(subtrahend) == h5py._hl.files.File:
        sub = np.array(subtrahend['data'][:,:,:])

    # check if inputs are h5-files
    if type(dataset) == h5py._hl.dataset.Dataset:
        data = np.array(dataset[:,:,:])
    if type(subtrahend) == h5py._hl.dataset.Dataset:
        sub = np.array(subtrahend[:,:,:])

    # check if inputs are numpy arrays
    if type(dataset) == np.ndarray:
        data = dataset
    if type(subtrahend) == np.ndarray:
        sub = subtrahend

    result = data - sub # doing the subtraction ########################

    # parsing output, if necessary
    # writing out if handling files
    if resultType == str and outputFileName != None:
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        outputFile = h5py.File(outputFileName)
        transplant_keys(dataset, outputFileName)
        outputFile['data'] = result
    # simple type check if only knwon formats where handled
    elif resultType == h5py._hl.files.File or resultType == h5py._hl.dataset.Dataset or resultType == np.ndarray:
        pass
    # throw erors with unkown data types
    else:
        raise TypeError("I do not know how to handle the input type: ", resultType)

    return result




@helping_tools.deprecated
def invert_c_axis(dataSetFileName):
    dataSetFile = h5py.File(dataSetFileName, 'a')
    oldData = dataSetFile['data']
    newData = np.zeros(oldData.shape)

    for i in range(oldData.shape[2]):
        newData[:,:,-i] = oldData[:,:,i]
    dataSetFile['data'][:,:,:] = newData[:,:,:]
    dataSetFile.close()


# based on some pre-existing code by Thomas Weber
@helping_tools.deprecated
def combineThreeFrames(upperleft, upperright, lowerhalf, factor = 1):
    """
    Combines three frams in one frame, which is returned.
    The first frame covers the upper left, the second to upper right quadrant and the third the lower half
    The frames are shifted to avoid overlapping zero lines
    All frames are assumed to have the same shape
    """
    # make a type casting if neccessary
    if type(upperleft) == str:
        a = pixdata.pixdata()
        a.readYellFile(upperleft)
    else:
        a = upperleft
    if type(upperright) == str:
        b = pixdata.pixdata()
        b.readYellFile(upperright)
    else:
        b = upperright
    if type(lowerhalf) == str:
        c = pixdata.pixdata()
        c.readYellFile(lowerhalf)
    else:
        c = lowerhalf

    # adding the second to the upper right corner; note the offset to avoid overlapping line
    if a.dimension == 3:
        out = pixdata.pixdata(np.zeros((a.npixels[0] + 2, a.npixels[1] + 2, a.npixels[2])))
        out.axismin = a.axismin
        out.pixelsize = a.pixelsize
        out.data[0:a.npixels[0] // 2 + 1, a.npixels[1] // 2 + 2:a.npixels[1] + 2, 0:a.npixels[2] ] = \
                 a.data[0:a.npixels[0] // 2 + 1, a.npixels[1] // 2:a.npixels[1], 0:a.npixels[2] ]
        out.data[b.npixels[0] // 2 + 2: b.npixels[0] + 2, b.npixels[1] // 2 + 2: b.npixels[1] + 2, 0:out.npixels[2]] = \
                 b.data[b.npixels[0] // 2: c.npixels[0], b.npixels[1] // 2: c.npixels[1], 0:c.npixels[2]]
        out.data[0:c.npixels[0] // 2 + 1, 0: c.npixels[1]//2 + 1, 0:c.npixels[2]] = \
                 c.data[0:c.npixels[0] // 2 + 1, 0:c.npixels[1] // 2 + 1, 0:c.npixels[2]]
        # ins the second half of the third multiplied by factor
        out.data[c.npixels[0] // 2 + 2: c.npixels[0] + 2, 0: c.npixels[1]//2 + 1, 0:c.npixels[2]] = \
                 c.data[c.npixels[0] // 2: c.npixels[0], 0:c.npixels[1] // 2 + 1, 0:c.npixels[2]] * factor
    else:
        out = pixdata.pixdata(np.zeros((a.npixels[0] + 2, a.npixels[1] + 2)))
        out.axismin = a.axismin
        out.pixelsize = a.pixelsize
        out.data[0:a.npixels[0] // 2 + 1, a.npixels[1] // 2 + 2:a.npixels[1] + 2 ] = \
                 a.data[0:a.npixels[0] / 2 + 1, a.npixels[1] // 2:a.npixels[1] ]
        out.data[b.npixels[0] // 2 + 2: b.npixels[0] + 2, b.npixels[1] // 2 + 2: b.npixels[1] + 2] = \
                 b.data[b.npixels[0] / 2: c.npixels[0], b.npixels[1] // 2: c.npixels[1]]
        out.data[0:c.npixels[0] // 2 + 1, 0: c.npixels[1]//2 + 1] = \
                 c.data[0:c.npixels[0] / 2 + 1, 0:c.npixels[1] // 2 + 1 ]
        # ins the second half of the third multiplied by factor
        out.data[c.npixels[0] // 2 + 2: c.npixels[0] + 2, 0: c.npixels[1]//2 + 1] = \
                 c.data[c.npixels[0] // 2: c.npixels[0], 0:c.npixels[1] // 2 + 1] * factor
    return out



# def add_metadata(pathToYellModels, pathToReciprocalDataSet, pathToPdfDataSet):
#     yellReciprocalModels = ["model.h5", "full.h5", "average.h5"]
#     yellPdfModels = ["delta-pdf.h5"]

#     # coping keys for reciprocal space data
#     print("Copying keys for reciprocal space data...")
#     for item in yellReciprocalModels:
#         transplant_keys(pathToReciprocalDataSet, os.path.join(pathToYellModels, item), False)

#     # coping keys for pdf space data
#     print("Copying keys for pdf space data...")
#     for item in yellPdfModels:
#         transplant_keys(pathToPdfDataSet, os.path.join(pathToYellModels, item), False)




# def get_atoms_by_distance(DistanceFilePath, uvw, threshold):
#     distance = DistanceBetween([0, 0, 0], uvw, GiveMeALattice())
#     print(distance)
#     with open(DistanceFilePath, 'r') as allDistances:
#         for line in allDistances:
#             atoms, currentDistance = line.split(',')
#             if abs(float(currentDistance) - distance) < threshold:
#                 print(line)


def GiveMeALattice():
    return np.array([19.328, 19.328, 28.858])

def TrigonalToCartesian(vector3d):
    alpha = np.deg2rad(30)
    return np.array([vector3d[0] - vector3d[1] * np.sin(alpha), vector3d[1] * np.cos(alpha), vector3d[2]])

def DistanceBetween(p1, p2, lattice):
    origin = lattice * np.array(p1) # creates a deep copy and casts to array if necessary
    target = lattice * np.array(p2) # creates a deep copy and casts to array if necessary
    u, v, w = target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]
    vector = TrigonalToCartesian([u, v, w])
    d = np.linalg.norm(vector)
    return d



def imtransform_centered(im, transformation_matrix):
    offset = np.dot(np.eye(2) - transformation_matrix, np.array(im.shape) / 2)
    im[np.isnan(im)] = 0
    return scipy.ndimage.affine_transform(im, transformation_matrix, offset=offset, order=1, )


def hextransform(im):
    T = np.array([[1, 0], [-math.cos(math.pi / 3), -math.sin(math.pi / 3)]])
    return imtransform_centered(im, T)

def hextransform2(im):
    T = np.array([[1, 0], [math.cos(math.pi / 3), math.sin(math.pi / 3)]])
    return imtransform_centered(im, T)