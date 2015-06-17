class MeerkatMetaData:
    """A class which provides the metadata of meerkat"""
    
    #onblock defined data types meerkat provides
    dtype_NORMAL = "Standard meerkat format"
    dtype_DATA = "Simplified data format"
    #offblock
    
    @property
    def isStandard(self):
        """Returns true if the meerkat format is the classical with rebinned, etc, datasets and false otherwise"""
        return self.format == self.dtype_NORMAL
    
    def __init__(self, file):
        """Constructor, requires an opened meerkat file in the hdf format."""
        #onblock setting of the data format
        if 'data' in file:
            self.format = self.dtype_DATA
        elif 'rebinned_data' in file:
            self.format = self.dtype_NORMAL
        else:
            raise TypeError("Given file does not match any currently known meerkat format.")
        #offblock
        
        self.pixelDimensions = np.asarray(file['number_of_pixels']) #setting dimensions in pixel of the reconstructed file
        self.hklRankge = np.asarray(file['maxind']) #hkl range of the reconstructed file
        self.steps = np.asarray(file['step_size']) #step sizes (hkl per pixel) of the reconstructed file