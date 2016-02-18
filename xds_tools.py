from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import re
import numpy as np
import fabio
import meerkat_tools
import helping_tools
import h5py
import shutil
import os
import os.path


UPDATE_UNITCELL = "Option switch to update the unit cell in XDS."
UPDATE_FRAMENAMETEMPLATE = "Option switch to set a new frame path, string adding new path to this option="
UPDATE_DATARANGE = "Option switch to change the frame range, string adding new lower,upper range to this="
UPDATE_SPOTRANGE = "Option switch to change the spot range, string adding new lower,upper range to this="
UPDATE_BGRANGE = "Option switch to change the background range, string adding new lower,upper range to this="

def numericalSort(value):
    """Creates a natural sorting function for file names, just for a nice display"""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def bgFromH5(pathToFile, frameShape):
    """Turns the background h5-file created by Albula into a numpy array"""
    file = h5py.File(pathToFile, 'r')
    a = np.asarray(file['entry/data/data'])
    bg = a.reshape((frameShape[0], frameShape[1], 1))[:,:,0]
    return bg


def XDS_update(pathToXdsFiles, updateOptions):
    """Makes a backup of the old XDS.INP file and creates a new updated one.
    pathToXdsFiles ... string location of XDS input and output files
    updateOptions ... array[string] specifies which parts should be updated, see constants for possibilities
    """
    XDSPath = os.path.join(pathToXdsFiles, "XDS.INP")
    IDXREFPath = os.path.join(pathToXdsFiles, "IDXREF.LP")
    # make a backup of the old XDS.INP
    shutil.copy(XDSPath, os.path.join(pathToXdsFiles, "%s_XDS.INP" % helping_tools.timestamp()))
    print("Created backup of XDS.INP")
    # read the XDS.INP file into an array for later manipulations
    xds = [line.rstrip('\n') for line in open(XDSPath)]
    # going through the options
    for option in updateOptions:
        # the unit cell should be updated from the refinement
        if option == UPDATE_UNITCELL:
            print("Updating unit cell ...")
            # search for the old cell
            oldCellIndex = get_xds_index(xds, "UNIT_CELL_CONSTANTS")
            if oldCellIndex == None:
                raise IndexError("No unit cell entry found!")
            print("Old cell is %s" % xds[oldCellIndex].split('=')[1])

            # hunt the new unit cell
            with open(IDXREFPath) as file:
                for line in file:
                    if "UNIT CELL PARAMETERS" in line:
                        newCell = line
                        # no break, multiple UNIT CELL PARAMETERS passges in file, the last one is the accurate one
            newCell = " ".join(newCell.split()[3:]) # format for output
            print("New cell is %s" % newCell)
            newCell = "UNIT_CELL_CONSTANTS= %s !updated by xds_tools.py" % newCell
            xds[oldCellIndex] = newCell

        # change the name template and path
        if UPDATE_FRAMENAMETEMPLATE in option:
            print("Updating name template")
            newTemplate = option.split('=')[1]
            oldTemplateIndex = get_xds_index(xds, "NAME_TEMPLATE_OF_DATA_FRAMES")
            print("Old name template is %s" % xds[oldTemplateIndex].split('=')[1])
            newTemplate = "NAME_TEMPLATE_OF_DATA_FRAMES=%s ! updated by xds_tools.py" % newTemplate
            xds[oldTemplateIndex] = newTemplate
            print("New name template is %s" % newTemplate)

        # change the data range
        if UPDATE_DATARANGE in option:
            print("Updating data range")
            ranges = option.split('=')[1]
            # extract the new ranges
            lower = int(ranges.split(',')[0])
            upper = int(ranges.split(',')[1])
            oldRangeIndex = get_xds_index(xds, "DATA_RANGE")
            print("Old data range is %s" % xds[oldRangeIndex].split('=')[1])
            newRange = "DATA_RANGE= %s %s ! updated by xds_tools.py" % (lower, upper)
            xds[oldRangeIndex] = newRange
            print("New data range is %s %s" % (lower, upper))

        # change the data range
        if UPDATE_DATARANGE in option:
            print("Updating data range")
            ranges = option.split('=')[1]
            # extract the new ranges
            lower = int(ranges.split(',')[0])
            upper = int(ranges.split(',')[1])
            oldRangeIndex = get_xds_index(xds, "DATA_RANGE")
            print("Old data range is %s" % xds[oldRangeIndex].split('=')[1])
            newRange = "DATA_RANGE= %s %s ! updated by xds_tools.py" % (lower, upper)
            xds[oldRangeIndex] = newRange
            print("New data range is %s %s" % (lower, upper))

        # change the data range
        if UPDATE_SPOTRANGE in option:
            print("Updating spot range")
            ranges = option.split('=')[1]
            # extract the new ranges
            lower = int(ranges.split(',')[0])
            upper = int(ranges.split(',')[1])
            oldRangeIndex = get_xds_index(xds, "SPOT_RANGE")
            print("Old spot range is %s" % xds[oldRangeIndex].split('=')[1])
            newRange = "SPOT_RANGE= %s %s ! updated by xds_tools.py" % (lower, upper)
            xds[oldRangeIndex] = newRange
            print("New spot range is %s %s" % (lower, upper))

        # change the data range
        if UPDATE_BGRANGE in option:
            print("Updating background range")
            ranges = option.split('=')[1]
            # extract the new ranges
            lower = int(ranges.split(',')[0])
            upper = int(ranges.split(',')[1])
            oldRangeIndex = get_xds_index(xds, "BACKGROUND_RANGE")
            print("Old data range is %s" % xds[oldRangeIndex].split('=')[1])
            newRange = "BACKGROUND_RANGE= %s %s ! updated by xds_tools.py" % (lower, upper)
            xds[oldRangeIndex] = newRange
            print("New background range is %s %s" % (lower, upper))
    # wirte the new XDS file
    print("Writing new XDS.INP ...")
    os.remove(XDSPath)
    XDSFile = open(XDSPath, 'w')
    for line in xds:
        XDSFile.write("%s\n" % line)
    print("Done!")


def get_xds_index(xds, entry):
    """Read out the zero based line number of an entry in XDS.INP.
    xds ... array[string] the xds input file, each entry in the array represents a line
    entry ... string should be found in the xds input file
    returns int index in the xds array which contains the entry, returns None if not found
    """
    index = None
    for i in range(len(xds)):
        if entry in xds[i]:
            index = i
            break # index is found, abort loop
    return index


def copy_xds_template(pathToXds, pathToDestination):
    """Copies a xds template input file to a new location.
    pathToXds ... string path to the xds file, file name including, will be renamed to XDS.INP.
    pathToDestination ... string path were the new copy should be placed
    """
    shutil.copy(pathToXds, os.path.join(pathToDestination, "XDS.INP"))
    print("Copied the XDS template to %s" % pathToDestination)
    # I'm not exactly sure why I need the function