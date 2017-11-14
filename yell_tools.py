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
import os.path


def make_array_fftready(data):
    """
    Cuts off the last indices of an odd array in order to prevent sepctral leakage during the FFT. Works with 2D and 3D arrays.
    data ... numpy.array The array which is to be pruned.
    returns ... numpy.array The modified array without the odd indices.
    """
    result = None
    if len(data.shape) == 2:
        if (data.shape[0] % 2 == 1) and (data.shape[1] % 2 == 1):   # only do something if all array indices are odd
            result = np.array(data[:-1, 1:]) # deep copy of the result
        else:
            raise IndexError("Please provide an array where all indices are odd!")
    elif len(data.shape) == 3:
        if (data.shape[0] % 2 == 1) and (data.shape[1] % 2 == 1) and (data.shape[2] % 2 == 1):   # only do something if all array indices are odd
            result = np.array(data[:-1, 1:, 1:]) # deep copy of the result
        else:
            raise IndexError("Please provide an array where all indices are odd!")
    else:
        raise IndexError("Input array has an unkown shape!")

    return result


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
        keysTo[item] = np.array(keysFrom[item])
    # clean up
    keysFrom.close()
    keysTo.flush()
    keysTo.close()


def add_metadata(pathToYellModels, pathToReciprocalDataSet, pathToPdfDataSet):
    yellReciprocalModels = ["model.h5", "full.h5", "average.h5"]
    yellPdfModels = ["delta-pdf.h5"]

    # coping keys for reciprocal space data
    print("Copying keys for reciprocal space data...")
    for item in yellReciprocalModels:
        transplant_keys(pathToReciprocalDataSet, os.path.join(pathToYellModels, item), False)

    # coping keys for pdf space data
    print("Copying keys for pdf space data...")
    for item in yellPdfModels:
        transplant_keys(pathToPdfDataSet, os.path.join(pathToYellModels, item), False)



def CrystalMaker_to_csv(InputFilePath, OutputFilePath):
    #helping_tools.check_folder(OutputFilePath)
    with open(InputFilePath, 'r') as inputFile:
        with open(OutputFilePath, 'w') as outputFile:
            for line in inputFile:
                parts = " ".join(line.split()).split(" ")
                atoms = parts[4] + "-" + parts[6]
                distance = parts[7]
                outputFile.writelines(",".join([atoms, distance]))
                outputFile.writelines("\n")


def get_atoms_by_distance(DistanceFilePath, uvw, threshold):
    distance = DistanceBetween([0, 0, 0], uvw, GiveMeALattice())
    print(distance)
    with open(DistanceFilePath, 'r') as allDistances:
        for line in allDistances:
            atoms, currentDistance = line.split(',')
            if abs(float(currentDistance) - distance) < threshold:
                print(line)


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


def CreateDistances(coordinateFile, cellExpansion):
    atomList = np.genfromtxt(coordinateFile, delimiter=',')
    distanceList = []
    xx = []
    yy = []
    zz = []
    for i in range(cellExpansion):
        for entry in atomList:
            for distanceTo in atomList:
                line = "%s-%s"


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