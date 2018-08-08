from __future__ import print_function # python 2.7 compatibility
from __future__ import division # python 2.7 compatibility

import sys
sys.path.append("/cluster/home/hoferg/python/lib64/python2.7/site-packages")
sys.path.append("/cluster/home/hoferg/python/lib64/python3.3/site-packages")

import numpy as np
import sympy as sp
import helping_tools
import yell_tools

# these are the transforamtion vectors
hex2cartesian = np.array([[1, -1/2, 0], [0, np.sqrt(3)/2, 0], [0,0,1]])
cartesian2hex = np.array([[1,np.sqrt(3)/3,0], [0, 2/np.sqrt(3),0],[0,0,1]])
FragmentNames = ["MS", "MT", "PS", "PT", "te"]

ORDER_xxyyzz_xyxzyz = "1"
ORDER_xxyyzz_xyyzxz = "2"
ORDER_xxyyzz_yzxzxy = "3" # this is the same order as shelx!
ORDER_shelx = "3"

def convert_tensor_hex_to_cartesian(U):
    """
    Transforms a tesnor described in the hexagonal lattice system into cartesian coordinates.
    U ... np.array[3,3] a tensor given in the hexagonal lattice system that should be transforemed into cartesian coordinates.
    returns ... np.array the transforemed tensor.
    """
    return np.dot(np.dot(hex2cartesian,U),np.transpose(hex2cartesian))


def convert_tensor_cartesian_to_hex(U):
    """
    Transforms a tesnor described in cartesian coordinates into the hexagonal lattice system.
    U ... np.array[3,3] a tensor given in carthesian coordinates that should be transforemed into the hexagonal lattice system.
    returns ... np.array the transforemed tensor.
    """
    return np.dot(np.dot(cartesian2hex,U),np.transpose(cartesian2hex))


def rotate_tensor_cartesian_z(U, theta):
    """
    Rotates a 3x3 cartesion tensor along the z-axis.
    U ... np.array[3,3] a tensor that should be rotated
    theta ... float rotation in degree, theta > 0 counter-clock-wise
    retruns ... np.array of the rotated tensor
    """
    theta = np.deg2rad(theta)
    if U.shape == (2,2):
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif U.shape == (3,3):
        rotation = np.array([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]])
    else:
        raise IndexError("Shape ", U.shape, " is incomaptible with rotation.")
    return np.dot(np.dot(rotation,U),np.transpose(rotation))


def convert_vector_hex_to_cartesian(v):
    """
    Transforms a vector described in the hexagonal lattice system into cartesian coordinates.
    v ... np.array[3] a vector given in the hexagonal lattice system that should be transforemed into cartesian coordinates.
    returns ... np.array the transforemed tensor.
    """
    return np.dot(hex2cartesian,v)


def convert_vector_cartesian_to_hex(v):
    """
    Transforms a tesnor described in cartesian coordinates into the hexagonal lattice system.
    v ... np.array[3,3] a tensor given in carthesian coordinates that should be transforemed into the hexagonal lattice system.
    returns ... np.array the transforemed tensor.
    """
    return np.dot(cartesian2hex,v)


def rotate_vector_cartesian_z(v, theta):
    """
    Rotates a vector cartesion tensor along the z-axis.
    U ... np.array[3] a vector that should be rotated
    theta ... float rotation in degree, theta > 0 counter-clock-wise
    retruns ... np.array of the rotated vector
    """
    theta = np.deg2rad(theta)
    if v.shape == (2,):
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif v.shape == (3,):
        rotation = np.array([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]])
    else:
        raise IndexError("Shape ", v.shape, " is incomaptible with rotation.")
    return np.dot(rotation,v)


def to_tensor(array, order=None):
    """
    Creates a 3x3 tensor from a given row vector representation of that tensor.
    array ... np.array with six elements
    order ... ORDER constant from this file which provides the element order in the given array.
    returns ... np.array a 3x3 tensor with elements x_ij = x_ji
    """
    T = np.zeros([3,3])
    if order == ORDER_xxyyzz_xyyzxz:
        T[0,0] = array[0]
        T[1,1] = array[1]
        T[2,2] = array[2]
        T[0,1] = array[3]
        T[1,0] = array[3]
        T[0,2] = array[5]
        T[2,0] = array[5]
        T[1,2] = array[4]
        T[2,1] = array[4]
    elif order == ORDER_xxyyzz_xyxzyz:
        T[0,0] = array[0]
        T[1,1] = array[1]
        T[2,2] = array[2]
        T[0,1] = array[3]
        T[1,0] = array[3]
        T[0,2] = array[4]
        T[2,0] = array[4]
        T[1,2] = array[5]
        T[2,1] = array[5]
    elif order == ORDER_xxyyzz_yzxzxy:
        T[0,0] = array[0]
        T[1,1] = array[1]
        T[2,2] = array[2]
        T[0,1] = array[5]
        T[1,0] = array[5]
        T[0,2] = array[4]
        T[2,0] = array[4]
        T[1,2] = array[3]
        T[2,1] = array[3]
    elif order == ORDER_shelx:
        T[0,0] = array[0]
        T[1,1] = array[1]
        T[2,2] = array[2]
        T[0,1] = array[5]
        T[1,0] = array[5]
        T[0,2] = array[4]
        T[2,0] = array[4]
        T[1,2] = array[3]
        T[2,1] = array[3]
    else:
        raise KeyError("Error in converting array to tensor: Ordering scheme not found or none given")
    return T


def real_space_metric_tensor(cell):
    """
    Calculates the real space matrix from a given cell
    cell ... np.array unit cell in row vector representation (a,b,c,alpha,beta,gamma).
    """
    G = np.zeros([3,3])
    G[0,0] = cell[0]**2
    G[1,1] = cell[1]**2
    G[2,2] = cell[2]**2
    G[0,1] = cell[0] * cell[1] * np.cos(np.deg2rad(cell[5]))
    G[1,0] = G[0,1]
    G[0,2] = cell[0] * cell[2] * np.cos(np.deg2rad(cell[4]))
    G[2,0] = G[0,2]
    G[1,2] = cell[1] * cell[2] * np.cos(np.deg2rad(cell[3])) # be careful with array notation, this is matrix element U23!!!
    G[2,1] = G[1,2]
    return G


def reciprocal_space_metric_tensor(cell):
    real = real_space_metric_tensor(cell)
    return np.linalg.inv(real)


def U_to_B(U):
    """
    Calculates the crystallographic B tensor from an U tensor.
    U ... np.ndarray anisotropic displacement paramters.
          Supports row vector (U11,U22,U33,U12,U13,U23) and matrix representation.
    returns ... np.ndarray the transformed B values in tensor format.
    """
    if U.shape == (6,):
        U = to_tensor(U,ORDER_xxyyzz_xyxzyz)
    elif U.shape == (3,3):
        pass # normal expected case so there is nothing to do here
    else:
        raise ValueError("Cannot parse input!")
    B = U * 8 * np.pi**2
    return B


def B_to_beta(B, cell):
    # B_ij = 4*beta_ij/(a*_i*a*_j)
    beta = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            beta[i,j] = B[i,j] * (1 / cell[j]) * (1 / cell[i]) / 4
    return beta


def create_6d_tensor(upperLeft, upperRight, lowerRight):
    """
    Creates a 6d, symmetrical along the main diagonal, tensor.
    All input arrays should be similar shaped square matrices.
    upperLeft ... np.ndarray upper left block matrix of the larger tensor.
    upperRight ... np.ndarray upper right block matrix of the larger tensor.
    lowerRight ... np.ndarray lower right block matrix of the larger tensor.
    returns ... np.ndarray the extended and symmetrized tensor.
    """
    print("Warning: create_6d_tensor only supports 3x3 matrices.")
    tensor = np.zeros([6,6])
    tensor[0:3,0:3] = upperLeft
    tensor[3:6,3:6] = lowerRight
    tensor[0:3,3:6] = upperRight
    return symmetrize(tensor)


def symmetrize(matrix):
    """
    Makes a matrix symmetrical along the main diagonal.
    matrix ... np.ndarray a square matrix that should be symmetrized
    returns ... np.ndarray the symmetrized matrix.
    """
    matrix = np.triu(matrix) # making sure that the lower lower triangular is all zero
    # if above is not met, the algorithm below will flip out
    return matrix + matrix.T - np.diag(matrix.diagonal())


def is_positive_definite(matrix):
    """
    Calculates wether a given matrix is positive definite.
    matrix ... np.ndarray matrix of an arbitrary shape.
    returns ... bool True if matrix is positive definite, false otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_positive_semidefinite(matrix):
    """
    Calculates wether a given matrix is positive semidefinite.
    matrix ... np.ndarray matrix of an arbitrary shape.
    returns ... bool True if matrix is positive semidefinite, false otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) >= 0)


def calculate_eigen_6d(UA, UB, UAB, hexToCartesian=True, precission=4, UABModifier=None):
    """
    Creates a six dimensional positive definite tensor and calculates its eigenvalues and eigenvectors.
    UA ... np.array[3,3] a three by three positive definite tensor to create the upper left blockmatrix of the 6D tensor.
    UB ... np.array[3,3] a three by three positive definite tensor to create the lower right blockmatrix of the 6D tensor.
    UAB ... np.array[3,3] a three by three matrix to create the upper right blockmatrix of the 6D tensor.
    hexToCartesian ... bool identifier if the input should undergo a transformation from the hexagonal to the cartesian lattice.
    precission ... int round the resulting eigenvalues and -vectors to this number of decimals.
    UABModifier ... np.array[3,3] this matrix will be multiplied to the UAB tensor before all else, useful for playing around with the UAB12+UAB21 problem.
    returns ...
                np.array[6] the six eigenvalues
                np.array[6,6] the corresponding eigenvectors
                np.array[7] the largest eigenvalue and its corresponding eigenvector for a quick read.
    """
    # first, modify the UAB matrix before all else
    if not UABModifier is None:
            UAB = UAB * UABModifier
    # make the coordinate transformation
    if hexToCartesian:
        UA = convert_tensor_hex_to_cartesian(UA)
        UB = convert_tensor_hex_to_cartesian(UB)
        UAB = convert_tensor_hex_to_cartesian(UAB)
    tensor = create_6d_tensor(UA,UAB,UB)
    eigenvalues, eigenvectors = np.linalg.eig(tensor)
    eigenvalues = np.round(eigenvalues, precission)
    eigenvectors = np.round(eigenvectors, precission)

    largestEigenvalue = np.argmax(eigenvalues)
    largestEigenvector = eigenvectors[:,largestEigenvalue]
    copyOutput = str(eigenvalues[largestEigenvalue]) + ";" + np.array2string(largestEigenvector,separator=';')[1:-1]
    return eigenvalues, eigenvectors, copyOutput


def calculate_correlations(UA, UB, UAB):
    """
    Calculates the correlations from given covariances.
    UA ... np.array[3,3] first element of the corresponding covariances.
    UB ... np.array[3,3] second element of the corresponding covariances.
    UA ... np.array[3,3] covariances of the two elements.
    returns ... np.array[3,3] the calculated correlations.
    """
    correlations = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            correlations[i,j] = UAB[i,j] / np.sqrt(UA[i,i] * UB[j,j])
    return correlations


def calculate_P_matrix(UA, UB, UAB):
    """
    Calculates the P-matrix, i.e. the coupled movement of two units, from the corresponding Us and covariances.
    UA ... np.array[3,3] first element of the corresponding covariances.
    UB ... np.array[3,3] second element of the corresponding covariances.
    UA ... np.array[3,3] covariances of the two elements.
    returns ... np.array[3,3] the calculated P-matrix.
    """
    P = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            P[i,j] = UA[i,j] + UB[i,j] - UAB[i,j] - UAB[j,i]
    return P


def green0_ADPs():
    """
    Static function which simply returns the ADP of crystal green0 as tensors.
    """
    MS = to_tensor([0.022,0.023,0.030,0.00984,-0.00083,-0.00022], ORDER_xxyyzz_xyxzyz)
    MT = to_tensor([0.021,0.024,0.036,0.00984,-0.00230,-0.00280], ORDER_xxyyzz_xyxzyz)
    te = to_tensor([0.032,0.034,0.054,0.01770,0.00140,0.00560], ORDER_xxyyzz_xyxzyz)
    return MS, MT, te


def breakup_6d_vector(vector):
    return vector[:3], vector[3:]


def angle_between(v1, v2, deg=True):
    result = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if deg:
        result = np.rad2deg(result)
    return(result)


#def visual_cholesky_decomposition_2D():
#    """
#    Generates a sympy output for the cholesky decomposition in two dimensions.
#    """
#    a=np.array([11,12,21,22])
#    UA = np.array(sp.symbols('UA_{0} UA_{1} UA_{1} UA_{3}'.format(*a))).reshape(2,2)
#    UB = np.array(sp.symbols('UB_{0} UB_{1} UB_{1} UB_{3}'.format(*a))).reshape(2,2)
#    UAB = np.array(sp.symbols('UAB_{0} UAB_{1} UAB_{2} UAB_{3}'.format(*a))).reshape(2,2)
#    P = sp.Matrix(np.zeros([4,4]))
#    P[0:2,0:2] = UA
#    P[2:4,2:4] = UB
#    P[2:4,0:2] = UAB
#    P[0:2,2:4] = UAB.T
#    #display(P)
#    display(P.cholesky())
#    #print(sp.latex(P.cholesky()))


def yell_parameter_list_to_tensor(parameterString, order=ORDER_xxyyzz_xyyzxz):
    """
    Creates a symmetrical tensor from a parameter definition list.
    parameterString ... string consecutive string of "variableName=value(error);" deliminted by a new line seperator.
    order ... covariance_tools.ORDER defines which part of the string corresponds to which matrix element.
    returns np.array[3,3] the complete tensor ready for calculations
    """
    parts = parameterString.split('\n')
    parts = list(filter(lambda a: a != '', parts)) # removes empty entries
    elements = []
    for p in parts:
        elements.append(p.split('=')[1].split(';')[0].split('(')[0]) # parse a line to a value
    return to_tensor(np.array(elements,dtype=float), order)


def yell_parameters_to_matrices(paramtersFileName):
    """
    Turns the refined ADP covariance parameters from yell into manageable matrices.
    paramtersFileName ... string file name and path to the file which contains ONLY the parameters.
                          Expects six consecutive parameters which make up the UAB matrix
                          a different number of consecutive parameters are discarded!
    returns ... list(covariance_tools.YellMatrices) a list where each entry is a complete set of covariance and correlation matrix from the refinement.
    """
    with open(paramtersFileName) as parametersFile:
        lines = parametersFile.readlines()
    # parameters that belong to the same UAB matrix are expected to differ only by their mode direction given as x, y or z
    modeFreeLines = [s.replace('x' , 'X') for s in lines]
    modeFreeLines = [s.replace('y' , 'X') for s in modeFreeLines]
    modeFreeLines = [s.replace('z' , 'X') for s in modeFreeLines]

    first = modeFreeLines[0]
    allMatrices = []
    currentBulk = []
    currentBulk.append(lines[0])
    for i in range(len(modeFreeLines) - 1):
        second = modeFreeLines[i + 1]
        if not (first.split('=')[0] == second.split('=')[0]):
            if len(currentBulk) == 6:
                A = YellMatrices("".join(currentBulk),UMatrices_green0(), locations_green0(), yell_tools.GiveMeALattice())
                A.calculate_matrices()
                allMatrices.append(A)
            currentBulk = []
        currentBulk.append(lines[i + 1])
        first = second
    return allMatrices


def UMatrices_green0():
    """
    Static method which holds the values of the U matrix for the refinement of green0.
    returns list(array[3,3]) the U tensors for a given fragment.
    """
    emptyArray = np.zeros([3,3])
    emptyArray[:,:] = np.nan
    MS = to_tensor((0.022251,0.023249,0.031520,-0.000237,-0.000983,0.009478),ORDER_xxyyzz_yzxzxy)
    MT = to_tensor((0.021328,0.024999,0.035395,-0.002278,-0.002044,0.010482), ORDER_xxyyzz_yzxzxy)
    PS = emptyArray
    PT = emptyArray
    te = to_tensor((0.032040,0.03429,0.053347,0.006710,0.002009,0.017843), ORDER_xxyyzz_yzxzxy)
    return [MS, MT, PS, PT, te]


def locations_green0():
    """
    Static method which returns the geometric center of each molecular fragment
    returns list[5](array[3]) ... a nested list-array which contains the xyz coordinates of all fragments.
    """
    MS = np.array([0.153607,0.2182747,0.856816111]) # MS
    MT = np.array([0.2171297,0.06003745,0.1410432]) # MT
    PS = None                                       # PS
    PT = None                                       # PT
    te = np.array([0.2230557,0.08524305,0.5559135]) # te
    return [MS, MT, PS, PT, te]


def Ruvws_from_file(filePath):
    """
    Generates nested lists where each list entry contains a again a list.
    The first entry is the Ruvw vector and the second is a list which contains all variables names that have this Ruvw vector.
    Input file must be formatted like:
    0,0,0
    variableName
    variableName
    filePath ... string location to the file which conatins the Ruvw relations
    returns ... list[np.array,list[string]] the information on all Ruvws
    """
    with open(filePath) as file:
        content = file.readlines()
    allRuvw = []
    entries = []
    r = None

    for line in content:
        if ',' in line:
            allRuvw.append([r,entries]) # if it is the first occurence, a dummy entry will be generated
            # Parsing Ruvw
            r = line.split(',')
            r = np.array([sp.sympify(r[0]).evalf(), sp.sympify(r[1]).evalf(), sp.sympify(r[2]).evalf()])
            entries = []
        else:
            entries.append(line.strip())
    # applying the last entry
    allRuvw.append([r,entries])

    del allRuvw[0] # first one is a dummy entry
    return allRuvw




class YellMatrices(object):
    """
    Holds the UAB, UA and UB matrix for a given fragment pair and calculates the relevant PV and PR matrices.
    """

    def __init__(self, UABString, UMatrices, locations, trigonalLattice):
        """
        Constructer, creates and calculates all relevant matrices from a given molecular fragment pair.
        UABString ... string new-line seperated string that holds the variables and values of a UAB block.
        UMatrices ... U matrices of all molecular fragments, constructor will figure out which belong to the given UAB matrix.
        locations ... np.array position of all molecular fragments, constructor will figure out which belong to the given UAB matrix.
        trigonalLattice ... np.array the a, b and c axis of a trigonal lattice
        """
        emptyArray = np.zeros([3,3])
        emptyArray[:,:] = np.nan
        # class variables
        descriptor = ""
        self.trigonalLattice = trigonalLattice
        self.rA = np.zeros([3,3])
        self.rB = np.zeros([3,3])
        self.Ruvw = np.nan
        self.UA = emptyArray.copy()
        self.UB = emptyArray.copy()
        self.distance = np.nan
        self.localVector = np.nan
        self.UAIdentifier = ""
        self.UBIdentifier = ""
        self.UAB = emptyArray.copy()
        self.PR = emptyArray.copy()
        self.PV = emptyArray.copy()
        self.cor = emptyArray.copy()
        self.PVeig = None
        self.V = np.zeros([6,6])
        self.V[:,:] = np.nan
        self.UABString = None
        self.PR0 = emptyArray.copy()
        self.PV0 = emptyArray.copy()
        self.UABString = UABString
        self.UAB = yell_parameter_list_to_tensor(self.UABString)
        self.descriptor = self.UABString.split('=')[0]
        self.orientations = -1

        # additional matrices for special properties and compare
        self.PVTranslationX = emptyArray.copy()
        self.PVTranslationY = emptyArray.copy()
        self.PVTranslationZ = emptyArray.copy()

        self.PVTranslationMinusX = emptyArray.copy()
        self.PVTranslationMinusY = emptyArray.copy()
        self.PVTranslationMinusZ = emptyArray.copy()

        # finding out which U matrices are belonging to this correlation
        fragments = []
        designators = np.array(["MS", "MT", "PS", "PT", "te"])
        # find returns -1 if the string is not found
        fragments.append(self.descriptor[0:6].find('MS'))
        fragments.append(self.descriptor[0:6].find('MT'))
        fragments.append(self.descriptor[0:6].find('PS'))
        fragments.append(self.descriptor[0:6].find('PT'))
        fragments.append(self.descriptor[0:6].find('te'))
        fragments = np.nonzero(np.array(fragments) + 1)[0] # read below
        # creates a mask for the relevant U matrices
        # nonzero returns a tuple, one entry for each array axis, since the array
        # is only 1D, the first tuple entry is selected

        # sorting the U matrices into their correct positions
        if len(fragments) == 1:
            self.UA = UMatrices[fragments[0]]
            self.UB = self.UA
            self.rA = locations[fragments[0]]
            self.rB = self.rA
            self.UAIdentifier = FragmentNames[fragments[0]]
            self.UBIdentifier = self.UAIdentifier
        elif len(fragments) == 2:
            # checking which parts comes first
            selected = designators[fragments]
            if self.descriptor[0:6].find(selected[0]) < self.descriptor[0:6].find(selected[1]):
                self.UA = UMatrices[fragments[0]]
                self.UB = UMatrices[fragments[1]]
                self.rA = locations[fragments[0]]
                self.rB = locations[fragments[1]]
                self.UAIdentifier = FragmentNames[fragments[0]]
                self.UBIdentifier = FragmentNames[fragments[1]]
            else:
                self.UA = UMatrices[fragments[1]]
                self.UB = UMatrices[fragments[0]]
                self.rA = locations[fragments[1]]
                self.rB = locations[fragments[0]]
                self.UAIdentifier = FragmentNames[fragments[1]]
                self.UBIdentifier = FragmentNames[fragments[0]]
        else:
            raise IndexError("More matrix identifiers in descriptive string than expected!")

        # getting the orientations of the fragment
        parts = self.descriptor.split('_')
        for p in parts:
            if p.isdigit():
                number = int(p)
                if number >= 11: # check whether it really is the orientations identifier
                    self.orientations = str(number)
        if self.orientations == -1:
            print("Warning: no orientation given for ", self.descriptor)
        else:
            # rotating the U matrices and locations if necessary
            # for point fragment A
            if int(self.orientations[0]) >= 2:
                t = convert_tensor_hex_to_cartesian(self.UA)
                t = rotate_tensor_cartesian_z(t,120)
                self.UA = convert_tensor_cartesian_to_hex(t)

                self.rA = convert_vector_hex_to_cartesian(self.rA)
                self.rA = rotate_vector_cartesian_z(self.rA, 120)
                self.rA = convert_vector_cartesian_to_hex(self.rA)
            if int(self.orientations[0]) >= 3:
                t = convert_tensor_hex_to_cartesian(self.UA)
                t = rotate_tensor_cartesian_z(t,120)
                self.UA = convert_tensor_cartesian_to_hex(t)

                self.rA = convert_vector_hex_to_cartesian(self.rA)
                self.rA = rotate_vector_cartesian_z(self.rA, 120)
                self.rA = convert_vector_cartesian_to_hex(self.rA)
            # for point fragment B
            if int(self.orientations[1]) >= 2:
                t = convert_tensor_hex_to_cartesian(self.UB)
                t = rotate_tensor_cartesian_z(t,120)
                self.UB = convert_tensor_cartesian_to_hex(t)

                self.rB = convert_vector_hex_to_cartesian(self.rB)
                self.rB = rotate_vector_cartesian_z(self.rB, 120)
                self.rB = convert_vector_cartesian_to_hex(self.rB)
            if int(self.orientations[1]) >= 3:
                t = convert_tensor_hex_to_cartesian(self.UB)
                t = rotate_tensor_cartesian_z(t,120)
                self.UB = convert_tensor_cartesian_to_hex(t)

                self.rB = convert_vector_hex_to_cartesian(self.rB)
                self.rB = rotate_vector_cartesian_z(self.rB, 120)
                self.rB = convert_vector_cartesian_to_hex(self.rB)
            self.distance = yell_tools.DistanceBetween(self.rA, self.rB, self.trigonalLattice)


    def calculate_matrices(self):
        # calculating the PV matrices

        self.PV0 = self.UA + self.UB

        for i in range(3):
            for j in range(3):
                self.PV[i,j] = self.PV0[i,j] - self.UAB[i,j] - self.UAB[j,i]
        self.PVeig = np.linalg.eig(self.PV)

        # calculating the PR matrices
        for i in range(3):
            for j in range(3):
                self.PR0[i,j] = self.PV0[i,j] / np.sqrt(self.PV0[i,i] * self.PV0[j,j])
        for i in range(3):
            for j in range(3):
                self.PR[i,j] = self.PV[i,j] / np.sqrt(self.PV[i,i] * self.PV[j,j])

        # calculating the correlation
        for i in range(3):
            for j in range(3):
                self.cor[i,j] = self.UAB[i,j] / np.sqrt(self.UA[i,i] * self.UB[j,j])

        # calculating the V matrix
        self.V[0:3,0:3] = self.UA
        self.V[3:6,3:6] = self.UB
        self.V[3:6,0:3] = self.UAB
        self.V[0:3,3:6] = self.UAB.T

        # checking the V matrix
        eigenvalues = np.linalg.eigvals(self.V)
        if len(eigenvalues[eigenvalues <= 0]) > 0:
            print("Warning: negative eigenvalues in ", self.descriptor)


        # calculate additional matrices ******************************
        # positive translation correlation in x
        hypotheticalCorrelation = np.array([[1,0,0], [0,0,0], [0,0,0]])
        self.PVTranslationX = self.get_hypothetical_PV(hypotheticalCorrelation)
        hypotheticalCorrelation = np.array([[0,0,0], [0,1,0], [0,0,0]])
        self.PVTranslationY = self.get_hypothetical_PV(hypotheticalCorrelation)
        hypotheticalCorrelation = np.array([[0,0,0], [0,0,0], [0,0,1]])
        self.PVTranslationZ = self.get_hypothetical_PV(hypotheticalCorrelation)

        hypotheticalCorrelation = np.array([[-1,0,0], [0,0,0], [0,0,0]])
        self.PVTranslationMinusX = self.get_hypothetical_PV(hypotheticalCorrelation)
        hypotheticalCorrelation = np.array([[0,0,0], [0,-1,0], [0,0,0]])
        self.PVTranslationMinusY = self.get_hypothetical_PV(hypotheticalCorrelation)
        hypotheticalCorrelation = np.array([[0,0,0], [0,0,0], [0,0,-1]])
        self.PVTranslationMinusZ = self.get_hypothetical_PV(hypotheticalCorrelation)



    def get_hypothetical_PV(self, hypotheticalCorrelation):
        hypotheticalPV = np.zeros([3,3])
        hypotheticalUAB = np.zeros([3,3])

        for i in range(3):
            for j in range(3):
                hypotheticalUAB[i,j] = hypotheticalCorrelation[i,j] * np.sqrt(self.UA[i,i] * self.UB[j,j])

        for i in range(3):
            for j in range(3):
                hypotheticalPV[i,j] = self.PV0[i,j] - hypotheticalUAB[i,j] - hypotheticalUAB[j,i]
        return hypotheticalPV


    def recalculate_distance(self, pathToRuvw):
        allRuvws = Ruvws_from_file(pathToRuvw)
        for item in allRuvws:
            if self.descriptor in item[1]:
                self.Ruvw = item[0]
                self.distance = yell_tools.DistanceBetween(self.rA,(self.rB + self.Ruvw).astype(float), self.trigonalLattice)
                self.localVector = self.rB + self.Ruvw - self.rA


    def max_PV(self):
        return np.max(self.PV)


    def max_PR(self):
        return np.max(self.PR)


    def __str__(self):
        outputString = "Providing statistics for " + self.descriptor
        outputString += "\nUA:\n" + str(self.UA)
        outputString += "\nUB:\n" + str(self.UB)
        outputString += "\nDistance:\n" + str(self.distance)
        outputString += "\nUAB:\n" + str(self.UAB)
        outputString += "\nCorrelation:\n" + str(self.cor)
        outputString += "\nPV_0:\n" + str(self.PV0)
        outputString += "\nPV:\n" + str(self.PV)
        outputString += "\nEigenvalues:\n" + str(self.PVeig[0])
        #outputString += "\nPR_0:\n" + str(self.PR0)
        #outputString += "\nPR:\n" + str(self.PR)
        #outputString += "\nV:\n" + str(self.V)
        #outputString += "\nEigenvalues:\n" + str(np.linalg.eigvals(self.V))
        return outputString
