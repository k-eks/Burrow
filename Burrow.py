import cmd, sys, io, os.path
import fancy_output as out
import analyze_data as ad
import meerkat_tools as mt
import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot
from matplotlib import colors as colorrange

class Burrow(cmd.Cmd):
    """Processing of data generated by meerkat"""

    #onblock "constructor" and "destructor"
    def preloop(self):
        """Setup of all data and settings for further use."""
        self.dset = []
        self.dsetName = []
        self.currentData = None
        self.meta = None
        self.activeDset = None
        self.currentImage = Image.new("RGB", (512,512)), "magenta"
        pyplot.ion() #turning interactive mode on
        self.contrast_min = 0
        self.contrast_max = 20
        self.cmaps = ['Greys', 'gist_rainbow']
        self.cmap_selection = 0
        print("Type \"help\" to get a list of commands.")

    def postloop(self):
        """Destructor, closes the hdf file."""
        if self.dset != None:
            for i in range(len(self.dset)):
                self.dset[i].close()
    #offblock

    @property
    def dataset(self):
        """Gets the active dataset."""
        return self.dset[self.dsetName.index(self.activeDset)]

    #onblock command line commands
    #onblock exit comands
    def do_exit(self, line):
        """exit the program"""
        return True

    def help_exit(self):
        """exit help page entry."""
        print("This commad exits the program.")

    def do_EOF(self, line):
        """This is the representation of the Ctrl+D shortcut."""
        return True

    def help_EOF(self):
        """EOF help page entry."""
        print("This is the representation of the Ctrl+D shortcut.")
        print("Hit Ctrl+D to exit the program.")
    #offblock

    def do_openFile(self, argument):
        """Opens a given file."""
        errorcode, arguments = self.getArg(argument)
        filename = "reconstruction.h5" #a default filename is assumed
        filealias = "default" #a default name is assumed
        if errorcode != -2:
            if "-n" in arguments:
                filename = arguments[arguments.index("-n") + 1]
            else:
                out.warn("No file name is given! Trying default name \"" + filename + "\".")
            if "-a" in arguments:
                filealias = arguments[arguments.index("-a") + 1]
            if os.path.isfile(filename):
                #TODO:requires replacement with arkadiy's routine
                file = h5py.File(filename, 'r')
                out.okay("File successfully opened as " + filealias + "!")
                self.activeDset = filealias
                pyplot.close("all") #to prevent display issues
                if filealias in self.dsetName:
                    self.dset[self.dsetName.index(filealias)] = file
                else:
                    self.dset.append(file)
                    self.dsetName.append(filealias)
                self.meta = mt.MeerkatMetaData(self.dataset)
                out.warn("Active data set is now " + filealias)
            else:
                out.error("File \"" + filename + "\" does not exist!")

    def help_openFile(self):
        """openFile help page entry."""
        print("Opens a h5 file, either in meerkat or direct format.")
        out.error("Only meerkat data format implemented at the moment!")
        print("Arguments:")
        print("\t-n <file name>\tspecifies the file name, if not given \"reconstruction.h5\" is assumed.")
        print("\t-a <file alias>\tspecifies the alias for the data under which it can be called , if not given \"default\" is assumed.")
    #offblock

    def do_showFiles(self, argument):
        """Output of all active dsets"""
        print("Active file: " + str(self.activeDset))
        print("Currently open files:")
        for s in self.dsetName:
            print(s)

    def help_showFiles(self):
        """showFiles help page entry"""
        print("Prints all of the currently active files.")

    def do_plothkl(self, argument):
        """plots a section of meerkat"""
        out.warn("Only poor error checking implemented!")
        errorcode, arguments = self.getArg(argument)
        if errorcode != -1 and errorcode != -2:
            index = 0 #default value
            if "-s" in arguments:
                section = arguments[arguments.index("-s") + 1]
            if "-i" in arguments:
                index = arguments[arguments.index("-i") + 1]

            trans = ad.Transformations(self.dataset, section)
            self.currentData, x = ad.crossection_data(self.dataset, float(index), trans)
            self.currentData = np.tile(self.currentData, (1,1))
            self.currentData = ad.hextransform(self.currentData)
            self.replot()
        elif errorcode == -1:
            out.error("No arguments supplied!")

    def help_plothkl(self):
        """plothkl help page entry."""
        print("Plots a section of the reciprocal space.")
        print("Arguments:")
        print("\t-s <section>\tselect a section, either hkx, hxl, xhl, uvx, uxw or xvw.")
        print("\t-i <index>\tfills the place holder of x with a Miller index, if none is given, 0 is assumed")

    def do_layout(self, argument):
        """Changes the layout of pyplot."""
        errorcode, arguments = self.getArg(argument)
        if self.activeDset != None:
            if errorcode != -1 and errorcode != -2:
                if "-min" in arguments:
                    self.contrast_min = int(arguments[arguments.index("-min") + 1])
                if "-max" in arguments:
                    self.contrast_max = int(arguments[arguments.index("-max") + 1])
                if "-c" in arguments:
                    self.cmap_selection = int(arguments[arguments.index("-c") + 1])
                    if self.cmap_selection >= len(self.cmaps) or self.cmap_selection < 0:
                        out.error("Unknown color map index.")
                        self.cmap_selection = 0
                        out.warn("Color map set to default.")
                self.replot()
            elif errorcode != -1:
                out.error("Input required!")
        else:
            out.error("No data selected!")

    def help_layout(self):
        """layout help page entry."""
        print("Changes the layout of the plot.")
        print("\t-min <number> sets the minimum value of contrast")
        print("\t-max <number> sets the maximun value of contrast")
        print("\t-c <index> changes the color map, type \"help colormap\" to get a list of available color maps")

    def help_colormaps(self):
        """Displays all color maps"""
        print("Available color maps:")
        for i, color in enumerate(self.cmaps):
            print(i, color)

    def do_setActive(self, argument):
        """Activates a dataset."""
        errorcode, arguments = self.getArg(argument)
        alias = "default" #default value
        if errorcode != -1 and errorcode != -2:
            if "-a" in arguments:
                alias = arguments[arguments.index("-a") + 1]
            else:
                out.warn("No arguments supplied, trying default.")
            if alias in self.dsetName:
                self.activeDset = alias
                self.meta = mt.MeerkatMetaData(self.dataset)
                pyplot.close("all") #in order to prevent display issues
                out.okay("Activated " + alias + "!")
            else:
                out.error(alias + " not found!")
        elif errorcode == -1:
            out.error("No arguments supplied!")

    def help_setActive(self):
        """setActive help page entry"""
        print("Changes the active dataset")
        print("\t-a <alias> alias of the dataset which should be activated")

    def do_saveData(self, argument):
        """Saves the last displayed image as a csv file."""
        errorcode, arguments = self.getArg(argument)
        if errorcode != -1 and errorcode != -2:
            if "-o" in arguments:
                outfile = arguments[arguments.index("-o") + 1]
                data = np.asarray(self.currentData)
                np.savetxt(outfile, data, delimiter=";")
                out.okay("Successfully saved as " + outfile + "!")
            else:
                out.error("No output name given!")

    def help_saveData(self):
        print("Saves the last displayed image as a csv file.")
        print("\t-o <filename> name of the output file")


    #onblock internal functions
    def getArg(self, line):
        """Splits the arguments and checks if the correct number of arguments are given."""
        errorcode = 0
        arguments = str(line).split()
        if len(arguments) == 0: #no arguments supplied
            arguments = ""
            errorcode = -1
        elif len(arguments) % 2 != 0: #invalid number of arguments
            arguments = ""
            errorcode = -2
            out.error("Invalid number of arguments!")
        return errorcode, arguments

    def replot(self):
        """pyplot distinguishes between 1D and 2D data, this function should call the right method."""
        self.plot()

    def plot(self):
        """Uses pyplot to draw 2D data."""
        pyplot.clf()
        height = self.meta.shape[0] #TODO:needs adjustment for different cuts
        width = self.meta.shape[1]  #TODO:needs adjustment for different cuts
        if (height + width > 1000): #here are some issues with the display size
            dpi = width / 10
        else:
            dpi = width
        pyplot.figure(figsize=(height/99.9,width/99.9), dpi=dpi) #creates a display mash-up, corrected below
        #The above line is a critical when it comes to image size
        pyplot.axes([0,0,1,1])
        pyplot.axis("off")
        pyplot.imshow(self.currentData, interpolation='nearest', clim=[self.contrast_min, self.contrast_max], cmap=self.cmaps[self.cmap_selection])
        self.currentImage = self.plot2img(pyplot.figure)
        #the following two lines remove the display mash up produced above
        pyplot.close(len(pyplot.get_fignums()) - 1)
        pyplot.close(len(pyplot.get_fignums()) - 1)
        pyplot.show()

    #onblock convertion of images und plots
    def plot2img(self, figure):
        """Converts a pyplot figure into a PIL image by the use of the buffer."""
        buf = io.BytesIO()
        pyplot.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)

    def saveCurrentImage(self, filename):
        """Saves the last image created/displayed."""
        try:
            self.currentImage.save(filename)
            out.okay(filename + " successfully saved!")
        except:
            raise
    #offblock
    #end of internal functions
    #offblock

if __name__ == '__main__':
    """Main loop creation, can run in terminal mode or script mode."""
    if len(sys.argv) == 2: #in this case, it is assumed that the user provides a file with a list of commands
        input = open(sys.argv[1], 'rt')
        try:
            #setting up silent script mode
            interpreter = Burrow(stdin=input)
            interpreter.use_rawinput = False #required for file input to read new lines etc correctly
            interpreter.prompt = "" #silent mode
            interpreter.cmdloop()
        finally:
            input.close()
    elif len(sys.argv) == 1: #plain old commandline
        interpreter = Burrow()
        interpreter.cmdloop()
    else:
        out.error("Wrong input, <none> or <script filename> expected!")