import re
import numpy as np
import fabio
import meerkat_tools
import h5py

# some properties
DEFECTIVE = -2
UNTRUSTED = -1
BG_MASK = -3
MASKED_BRAGG_INTENSITY = -999999
SIMPLE_UNTRUSTED = -666666

FULLMASK = "Masking setting which differntiates between untrusted, defective and hot pixel."
SIMPLEMASK = "Masking setting which uses a simplified mask from the BKGPIX frame."

def numericalSort(value):
    """Creates a natural sorting function for file names, just for a nice display"""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def bgFromH5(pathToFile, frameShape):
    """Turns the background h5-file created by Albula into a numpy array"""
    file = h5py.File(pathToFile, 'r')
    a = np.asarray(file['entry/data/data_000001'])
    bg = a.reshape((frameShape[0], frameShape[1], 1))[:,:,0]
    return bg


def generate_all_unwanted_pixel(bgFrame, hotStart):
    """This function reads out all pixels which do not belong to the data set into linear arrays."""
    """These linear arrays can be used for fancy indexing."""
    defectivePixel = []
    untrustedPixel = []
    hotPixel = []
    shapeX = bgFrame.data.shape[0]
    shapeY = bgFrame.data.shape[1]
    for x in range(shapeX):
        for y in range(shapeY):
            if bgFrame.data[x, y] == DEFECTIVE:
                defectivePixel.append(x * bgFrame.data.shape[1] + y)
            elif bgFrame.data[x, y] == UNTRUSTED:
                untrustedPixel.append(x * bgFrame.data.shape[1] + y)
            elif bgFrame.data[x, y] >= hotStart: # intensity at which hot pixel start
                hotPixel.append(x * bgFrame.data.shape[1] + y)
    return np.asarray(defectivePixel), np.asarray(untrustedPixel), np.asarray(hotPixel)


def read_all_unwanted_pixel(pathToBKGPIX, frameTemplate):
    """This function reads out all pixels from the BKGPIX which do not belong
    to the data set into a linear array."""
    """These linear arrays can be used for fancy indexing."""
    untrustedPixel = []
    shapeX = frameTemplate.data.shape[0]
    shapeY = frameTemplate.data.shape[1]
    bg = bgFromH5(pathToBKGPIX, frameTemplate.data.shape)
    for x in range(shapeX):
        for y in range(shapeY):
            if bg[x, y] == BG_MASK:
                untrustedPixel.append(x * frameTemplate.data.shape[1] + y)
    return np.asarray(untrustedPixel)


def restore_simple_pixel_mask(frame, untrusted):
    """Uses a linear array to change masked pixel to a new value."""
    data = frame.data.flatten()
    data[untrusted] = SIMPLE_UNTRUSTED
    return np.reshape(data, frame.data.shape)


def restore_pixel_mask(frame, defective, untrusted, hot):
    """Uses linear arrays to return masked pixel to their original masked value."""
    data = frame.data.flatten()
    data[defective] = DEFECTIVE
    data[untrusted] = UNTRUSTED
    data[hot] = DEFECTIVE
    return np.reshape(data, frame.data.shape)