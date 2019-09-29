## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv


# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self):
        return {
            'backA': {'msg': 'Image filename for Background A Color', 'default': None},
            'backB': {'msg': 'Image filename for Background B Color', 'default': None},
            'compA': {'msg': 'Image filename for Composite A Color', 'default': None},
            'compB': {'msg': 'Image filename for Composite B Color', 'default': None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut': {'msg': 'Image filename for Object Color', 'default': ['color.tif']},
            'alphaOut': {'msg': 'Image filename for Object Alpha', 'default': ['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn': {'msg': 'Image filename for Object Color', 'default': None},
            'alphaIn': {'msg': 'Image filename for Object Alpha', 'default': None},
            'backIn': {'msg': 'Image filename for Background Color', 'default': None},
        }

    def compositingOutput(self):
        return {
            'compOut': {'msg': 'Image filename for Composite Color', 'default': ['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if key in Matting._images:
            if "alpha" in key:
                Matting._images[key] = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
            if Matting._images[key] == True:
                success = True
            else:
                msg = "Read Failed"
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        foreground = np.zeros(Matting._images["compA"].shape)
        alpha = np.zeros(Matting._images["compA"].shape[:2])

        c0_r = Matting._images["compA"][:, :, 0]
        c0_g = Matting._images["compA"][:, :, 1]
        c0_b = Matting._images["compA"][:, :, 2]
        c1_r = Matting._images["compB"][:, :, 0]
        c1_g = Matting._images["compB"][:, :, 1]
        c1_b = Matting._images["compB"][:, :, 2]

        b0_r = Matting._images["backA"][:, :, 0]
        b0_g = Matting._images["backA"][:, :, 1]
        b0_b = Matting._images["backA"][:, :, 2]
        b1_r = Matting._images["backB"][:, :, 0]
        b1_g = Matting._images["backB"][:, :, 1]
        b1_b = Matting._images["backB"][:, :, 2]

        for row in  Matting._images["compA"].shape(0):
            for pixel in row:
                deltaC0_r, deltaC0_g, deltaC0_b =\
                    c0_r[row, pixel] - b0_r[row, pixel], c0_g[row, pixel] - b0_g[row, pixel], c0_b[row, pixel] - b0_b[row, pixel]
                deltaC1_r, deltaC1_g, deltaC1_b =\
                    c1_r[row, pixel] - b1_r[row, pixel], c1_g[row, pixel] - b1_g[row, pixel], c1_b[row, pixel] - b1_b[row, pixel]
                A = np.array([[1, 0, 0, -(b0_r[row, pixel])], [0, 1, 0, -(b0_g[row, pixel])], [0, 0, 1, -(b0_b[row, pixel])],
                                   [1, 0, 0, -(b1_r[row, pixel])], [0, 1, 0, -(b1_g[row, pixel])], [0, 0, 1, -(b1_b[row, pixel])]])
                b = np.array([deltaC0_r, deltaC0_g, deltaC0_b, deltaC1_r, deltaC1_g, deltaC1_b])

                #calculation
                pseudo_inverse = np.linalg.pinv(A)
                x = np.dot(pseudo_inverse, b)

                foreground[row, pixel] = np.array(x[0], x[1], x[2])
                alpha[row, pixel] = x[3]

        Matting._images["colOut"] = foreground
        Matting._images["alphaOut"] = alpha
        #########################################

        return success, msg

    def createComposite(self):
        """
success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        #########################################

        return success, msg