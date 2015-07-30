import numpy as np
import h5py

def get_index_selection(sectionSelector):
    """Gets the index from the x in the string array, i.e. a sectionSelector of hkx returns 2"""
    return sectionSelector.find("x")


def get_section_raw(file, pixel):
    """Returns a crosssection of hkx by pixel height."""
    return file['rebinned_data'][:, :, pixel] / file['number_of_pixels_rebinned'][:, :, pixel]


def is_hole(intensity):
    """Checks if the pixel of a frame was affected py the punching algorithm"""
    return intensity < -2


def is_untrusted(intensity):
    """Checks if the pixel of a frame is either defect or a gap"""
    return intensity < 0 and intensity >= -2


def to_Yell_data(inputFileName, outputFileName):
    inFile = h5py.File(inputFileName, 'r')
    outFile = h5py.File(outputFileName, 'w')
    meta = MeerkatMetaData(inFile)

    outFile.create_dataset('data', (meta.shape[0], meta.shape[1], meta.shape[2]))
    for i in range(meta.shape[2]):
        print("Using index %i of %i" % (i, meta.shape[2] - 1), end = "\r")
        section = np.nan_to_num(get_section_raw(inFile, i))
        outFile['data'][:,:,i] = section
        del section # freeing memory

    print("\nConversion done!")
    inFile.close()
    outFile.close()



class MeerkatMetaData:
    """A class which provides the metadata of meerkat"""

    #onblock defined data types meerkat provides
    dtype_NORMAL = "Standard meerkat format"
    dtype_DATA = "Simplified data format"
    #offblock


    @property
    def is_normal_type(self):
        """Returns true if the meerkat format is the classical with rebinned, etc, datasets and false otherwise"""
        return self.format == self.dtype_NORMAL

    def __init__(self, file):
        """Constructor, requires an opened meerkat file in the hdf format."""
        #onblock setting of the data format
        if 'data' in file:
            self.format = self.dtype_DATA
            self.shape = np.asarray(file['data'].shape)
            self.hklRange = None #not stored in this data format
            self.steps = None #not stored in this data format
        elif 'rebinned_data' in file:
            self.format = self.dtype_NORMAL
            self.shape = np.asarray(file['number_of_pixels']) #setting dimensions in pixel of the reconstructed file
            self.hklRankge = np.asarray(file['maxind']) #hkl range of the reconstructed file
            self.steps = np.asarray(file['step_size']) #step sizes (hkl per pixel) of the reconstructed file
        else:
            raise TypeError("Given file does not match any currently known meerkat format.")
        #offblock


    def get_display_dimensions(self, sectionSelector):
        """returns the image settings for pyplot to display an image correctly"""
        shape = np.delete(self.shape, get_index_selection(sectionSelector))
        width = shape[0]
        height = shape[1]
        if (height + width > 1000): #here are some issues with the display size
            dpi = width / 10
        else:
            dpi = width
        return width, height, dpi


class MeerkatSection:
    """A class which provides simple access to a meerkat section."""

    def __init__(self, file, pixel):
        self.data = np.asarray(get_section_raw(file, pixel))
        self.pixel = pixel

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Pixel:
    """This class represents a pixel as a workaround for multidimensional arrays."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y)


    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Voxel:
    """This class represents a voxel as a workaround for multidimensional arrays."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


    def __str__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + " z = " + str(self.z)


    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z