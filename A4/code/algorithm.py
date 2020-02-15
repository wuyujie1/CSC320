# CSC320 Fall 2019
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    row = source_patches.shape[0]
    col = source_patches.shape[1]
    k = len(f_heap[0][0])
    best_D = np.empty([row, col])
    rand = int(np.ceil(- np.log10(w) / np.log10(alpha)))
    if (odd_iteration):
        for i in range(row):
            for j in range(col):
                if (propagation_enabled):
                    for q in range(k):
                        dscore = []
                        if (odd_iteration):
                            # add self to list
                            if (i - 1 >= 0):
                                if (i + f_heap[i - 1][j][q][2][0] < row):
                                    dscore.append((f_heap[i - 1][j][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i - 1][j][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i - 1][j][q][2][
                                                                                                1]])))
                            if (j - 1 >= 0):
                                if (j + f_heap[i][j - 1][q][2][1] < col):
                                    dscore.append((f_heap[i][j - 1][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i][j - 1][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i][j - 1][q][2][
                                                                                                1]])))
                        else:
                            if (i + 1 < row):
                                if (i + f_heap[i + 1][j][q][2][0] < row):
                                    dscore.append((f_heap[i + 1][j][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i + 1][j][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i + 1][j][q][2][
                                                                                                1]])))
                            if (j + 1 < col):
                                if (j + f_heap[i][j + 1][q][2][1] < col):
                                    dscore.append((f_heap[i][j + 1][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i][j + 1][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i][j + 1][q][2][
                                                                                                1]])))
                        nl = nlargest(k, f_heap[i][j])
                        for elm in dscore:
                            if (elm[1] > nl[k - 1][0]):
                                if not tuple([elm[0][0], elm[0][1]]) in f_coord_dictionary[i][j]:
                                    f_coord_dictionary[i][j][tuple([elm[0][0], elm[0][1]])] = 0
                                    can = (elm[1], _tiebreaker.next(), elm[0])
                                    worst = heappushpop(f_heap[i][j], can)
                                    if f_coord_dictionary[i][j].has_key((worst[2][0], worst[2][1])):
                                        f_coord_dictionary[i][j].pop((worst[2][0], worst[2][1]))
                if (random_enabled):
                    for q in range(k):
                        u = []
                        dscore_random = []
                        c = 0
                        while ((alpha ** c) * w > 1):
                            R = [int(np.random.uniform(-1, 1) * (alpha ** c) * w),
                                 int(np.random.uniform(-1, 1) * (alpha ** c) * w)]
                            x = i + f_heap[i][j][q][2][0] + R[0]
                            if x < 0 or x >= row:
                                break
                            y = j + f_heap[i][j][q][2][1] + R[1]
                            if (y < 0):
                                break
                            elif (y >= col):
                                break
                            dscore_random.append(
                                ((x - i, y - j), -calculate_D(source_patches[i][j], target_patches[x][y])))
                            c += 1
                        nl = nlargest(k, f_heap[i][j])
                        for elm in dscore_random:
                            if (elm[1] > nl[k - 1][0]):
                                if not tuple([elm[0][0], elm[0][1]]) in f_coord_dictionary[i][j]:
                                    f_coord_dictionary[i][j][tuple([elm[0][0], elm[0][1]])] = 0
                                    can = (elm[1], _tiebreaker.next(), elm[0])
                                    worst = heappushpop(f_heap[i][j], can)
                                    if f_coord_dictionary[i][j].has_key((worst[2][0], worst[2][1])):
                                        f_coord_dictionary[i][j].pop((worst[2][0], worst[2][1]))
    else:
        for i in range(row - 1, -1, -1):
            for j in range(col - 1, -1, -1):
                if (propagation_enabled):
                    for q in range(k):
                        dscore = []
                        if (odd_iteration):
                            # add self to list
                            if (i - 1 >= 0):
                                if (i + f_heap[i - 1][j][q][2][0] < row):
                                    dscore.append((f_heap[i - 1][j][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i - 1][j][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i - 1][j][q][2][
                                                                                                1]])))
                            if (j - 1 >= 0):
                                if (j + f_heap[i][j - 1][q][2][1] < col):
                                    dscore.append((f_heap[i][j - 1][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i][j - 1][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i][j - 1][q][2][
                                                                                                1]])))
                        else:
                            if (i + 1 < row):
                                if (i + f_heap[i + 1][j][q][2][0] < row):
                                    dscore.append((f_heap[i + 1][j][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i + 1][j][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i + 1][j][q][2][
                                                                                                1]])))
                            if (j + 1 < col):
                                if (j + f_heap[i][j + 1][q][2][1] < col):
                                    dscore.append((f_heap[i][j + 1][q][2], -calculate_D(source_patches[i][j],
                                                                                        target_patches[
                                                                                            i + f_heap[i][j + 1][q][2][
                                                                                                0]][
                                                                                            j + f_heap[i][j + 1][q][2][
                                                                                                1]])))
                        nl = nlargest(k, f_heap[i][j])
                        for elm in dscore:
                            if (elm[1] > nl[k - 1][0]):
                                if not tuple([elm[0][0], elm[0][1]]) in f_coord_dictionary[i][j]:
                                    f_coord_dictionary[i][j][tuple([elm[0][0], elm[0][1]])] = 0
                                    can = (elm[1], _tiebreaker.next(), elm[0])
                                    worst = heappushpop(f_heap[i][j], can)
                                    if f_coord_dictionary[i][j].has_key((worst[2][0], worst[2][1])):
                                        f_coord_dictionary[i][j].pop((worst[2][0], worst[2][1]))
                if (random_enabled):
                    for q in range(k):
                        u = []
                        dscore_random = []
                        c = 0
                        while ((alpha ** c) * w > 1):
                            R = [int(np.random.uniform(-1, 1) * (alpha ** c) * w),
                                 int(np.random.uniform(-1, 1) * (alpha ** c) * w)]
                            x = i + f_heap[i][j][q][2][0] + R[0]
                            if x < 0 or x >= row:
                                break
                            y = j + f_heap[i][j][q][2][1] + R[1]
                            if (y < 0):
                                break
                            elif (y >= col):
                                break
                            dscore_random.append(
                                ((x - i, y - j), -calculate_D(source_patches[i][j], target_patches[x][y])))
                            c += 1
                        nl = nlargest(k, f_heap[i][j])
                        for elm in dscore_random:
                            if (elm[1] > nl[k - 1][0]):
                                if not tuple([elm[0][0], elm[0][1]]) in f_coord_dictionary[i][j]:
                                    f_coord_dictionary[i][j][tuple([elm[0][0], elm[0][1]])] = 0
                                    can = (elm[1], _tiebreaker.next(), elm[0])
                                    worst = heappushpop(f_heap[i][j], can)
                                    if f_coord_dictionary[i][j].has_key((worst[2][0], worst[2][1])):
                                        f_coord_dictionary[i][j].pop((worst[2][0], worst[2][1]))

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    f_heap = [[[] for jj in range(source_patches.shape[1])] for ii in range(source_patches.shape[0])]
    f_coord_dictionary = [[{} for jjj in range(source_patches.shape[1])] for iii in range(source_patches.shape[0])]

    s = np.nan_to_num(source_patches)
    t = np.nan_to_num(target_patches)
    for i in range(source_patches.shape[0]):
        for j in range(source_patches.shape[1]):
            for k in range(f_k.shape[0]):
                displacement = f_k[k][i][j]
                priority = -np.sum(np.sum((t[i + displacement[0], j + displacement[1]] - s[i][j]) ** 2, axis=1), axis=0)
                counter = _tiebreaker.next()

                heappush(f_heap[i][j], (priority, counter, displacement))
                f_coord_dictionary[i][j][tuple(displacement)] = 0
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    f_k = np.zeros([len(f_heap[0][0]), len(f_heap), len(f_heap[0]), 2])
    D_k = np.zeros([len(f_heap[0][0]), len(f_heap), len(f_heap[0])])
    for i in range(len(f_heap)):
        for j in range(len(f_heap[0])):
            for k in range(len(f_heap[0][0])):
                D_k[k][i][j] = -f_heap[i][j][k][0]

                f_k[k][i][j][0] = f_heap[i][j][k][2][0]
                f_k[k][i][j][1] = f_heap[i][j][k][2][1]
    f_k =f_k.astype(int)
    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    denoised = np.zeros(target.shape)
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)
    f_k = (make_coordinates_matrix(target.shape) + f_k).reshape((-1, 2))

    tmp = target[f_k[:, 0], f_k[:, 1]].reshape((-1, target.shape[0], target.shape[1], 3))
    e = np.exp(-(D_k ** 0.5 / h ** 2))
    zi = np.sum(e, axis=0)
    wei = e / zi

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            for k in range(len(f_heap[0][0])):
                denoised[i, j] += tmp[k, i, j] * wei[k, i, j]

    #############################################

    return denoised


#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################

def calculate_D(source_patches, target_patches):
    p1 = source_patches.flatten()
    p2 = target_patches.flatten()
    n = p1-p2
    n = n[~np.isnan(n)]
    return np.linalg.norm(n)


#############################################



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    i = make_coordinates_matrix(target.shape) + f
    rec_source = target[np.clip(i[:, :, 0], 0, target.shape[0] - 1), np.clip(i[:, :, 1], 0, target.shape[1] - 1)]
    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
