import math
import numpy as np
import cv2
from decimal import Decimal
from itertools import product
from time import time
from fractions import Fraction

"""
Class for calculating Convexity based on the article:
A Spatial Convexity Descriptor for Object Enlacement
https://doi.org/10.1007/978-3-030-14085-4_26
"""
class Convexity:

    def __init__(self, image: np.array, verbose: bool = False, logname: str = None, feature_image: str = None):
        """
        Initializes a new object to compute convexity
        :param image: input image
        :param verbose: huge additional output
        :param logname: logfile name for logging individual values
        :param feature_image: filename for feature image, png preferred
        """
        self.verbose = verbose
        self.logname = logname
        self.feature_image = feature_image
        if image is None:
            raise Exception('No image')
        self.f = image
        # making the four quads. individual values can be very big, Decimal type must be used
        self.quad1 = np.zeros(image.shape, dtype=Decimal)
        self.quad2 = np.zeros(image.shape, dtype=Decimal)
        self.quad3 = np.zeros(image.shape, dtype=Decimal)
        self.quad4 = np.zeros(image.shape, dtype=Decimal)
        # final array for phi values
        self.phi = np.zeros(image.shape, dtype=Decimal)

    @staticmethod
    def deg2vec(deg: float):
        """
        Calculates the closest integer vector to represent a degree of rotation, e.g. 45 -> (1,1)
        note that not all input rotations have the vector with integer values on the 10x10 grid
        grid size must kept low because computational cost is O(number of elements)
        :param deg:
        :return: closest degree to the input, and the integer vector as a simplified fraction (num, denom)
        """
        # make the sin and cos values of the desired input angle
        s, c = math.sin(np.deg2rad(deg)), math.cos(np.deg2rad(deg))
        # multiply by 10 to find closest integer vector
        # Fraction class simplifies the fraction automatically
        f = Fraction(int(np.round(s * 10)), int(np.round(c * 10)))
        # get the corrected angle that corresponds to this integer vector
        r = math.atan2(f.numerator, f.denominator)
        # convert it back to deg from rad
        d = np.rad2deg(r)
        return d, f

    def calcH(self, img, vec):
        v1, v2 = vec  # v1 = x (col) offset, v2 = y (row) offset, positive is downwards
        b = np.array([v1 * ri - v2 * ci for ri, ci in product(range(img.shape[0]), range(img.shape[1]))]).reshape(img.shape)
        scanline_sums = {key: np.sum(img * (b == key)).astype(Decimal) for key in np.unique(b)}
        return scanline_sums, b

    # v(x,y) is from the descartes lower right quad
    def rotsat(self, a, v=(1, 0), qname='N/A'):
        # corners for the inner points to add, starting from 0,0 - v, clockwise, e.g.:
        # o o 2 o o
        # 1 o o o c
        # o o o 3 o
        # o x o o o
        corners = np.array([[0, 0], [-v[0], -v[1]], [-v[0] - v[1], -v[1] + v[0]], [-v[1], v[0]]])
        dy = v[1] / v[0]
        # points for the triangle below the 'window'
        lower_triangle_pts = []
        line_pos = corners[1][1]
        for i in range(corners[1][0], 0):
            for j in range(corners[1][1], -1, -1):
                if j <= line_pos:
                    lower_triangle_pts += [[i, j]]
            line_pos += dy
        # rotated versions of the outer triangles
        rot90 = [[x[1] - v[0], -x[0] - v[1]] for x in lower_triangle_pts]
        rot180 = [[x[1] - v[0], -x[0] - v[1]] for x in rot90]
        rot270 = [[x[1] - v[0], -x[0] - v[1]] for x in rot180]
        all_outside_pts = np.array(lower_triangle_pts + rot90 + rot180 + rot270)
        # making a mask to find out remaining (inner) points
        offset = all_outside_pts.min(axis=0)
        window_size = all_outside_pts.max(axis=0) - all_outside_pts.min(axis=0) + [1, 1]
        mask = np.ones(window_size, dtype=np.uint8)
        # mask indices cannot be negative so we offset it
        mask_indices = all_outside_pts - offset
        mask[mask_indices[:, 0], mask_indices[:, 1]] = 0
        # inner point indices
        mask_pos_ind = np.where(mask > 0)
        # summed rotated area table
        asum = np.zeros(a.shape, dtype=Decimal)
        for r, c in product(range(a.shape[0]), range(a.shape[1])):
            # bool array to show if a corner is inside the image area
            corner_in = [0 <= r - corners[i][1] < a.shape[0] and 0 <= c + corners[i][0] < a.shape[1] for i in
                         range(len(corners))]
            c1 = asum[r - corners[1][1], c + corners[1][0]] if corner_in[1] else 0
            c3 = asum[r - corners[3][1], c + corners[3][0]] if corner_in[3] else 0
            c2 = asum[r - corners[2][1], c + corners[2][0]] if corner_in[2] and corner_in[1] and corner_in[3] else 0
            pos_cols = mask_pos_ind[0] + offset[0] + c
            # mask_pos_ind (y) is descartian, we have to negate
            pos_rows = -mask_pos_ind[1] - offset[1] + r
            inside_pts = int(a[r, c])
            # collect inner points
            for pr, pc in zip(pos_rows, pos_cols):
                if 0 <= pr < a.shape[0] and 0 <= pc < a.shape[1]:
                    inside_pts += a[pr, pc]
            asum[r, c] = c1 - c2 + c3 + int(inside_pts)
            if self.verbose:
                print('q: {} pt r:{} c:{} v: {} c1({},{}): {} c2({},{}): {} c3({},{}): {} ins:{} => {}'.format(
                    qname, r, c, v, corners[1][0], corners[1][1], c1, corners[2][0], corners[2][1], c2, corners[3][0], corners[3][1], c3, inside_pts,
                    asum[r][c]))
        return asum

    def compute(self, obj_a=1, obj_b=0, vec=(1, 0)):
        stt = time()
        a, b = self.f == obj_a, self.f == obj_b
        if a.sum() == 0 or b.sum() == 0:
            raise Exception('object A or B is empty a:{} b:{} asum:{} bsum:{}'.format(obj_a, obj_b, a.sum(), b.sum()))
        frac = Fraction(vec[1], vec[0])
        if frac.numerator != vec[1]:
            print('{},{} is simplified to {},{}'.format(vec[1], vec[0], frac.numerator, frac.denominator))
            vec = (frac.denominator, frac.numerator)
        pad_size = max(abs(vec[0]), abs(vec[1]))
        a_pad = np.pad(a, pad_size, 'constant')
        self.quad1 = self.rotsat(a_pad, vec, 'z1')[pad_size:-pad_size, pad_size:-pad_size]
        self.quad2 = np.rot90(self.rotsat(np.rot90(a_pad, 1), vec, 'z2'), -1)[pad_size:-pad_size, pad_size:-pad_size]
        self.quad3 = np.rot90(self.rotsat(np.rot90(a_pad, -1), vec, 'z3'), 1)[pad_size:-pad_size, pad_size:-pad_size]
        self.quad4 = np.rot90(self.rotsat(np.rot90(a_pad, 2), vec, 'z4'), -2)[pad_size:-pad_size, pad_size:-pad_size]
        self.phi = self.quad1 * self.quad2 * self.quad3 * self.quad4 * (b == 1)
        card_f = a.astype(Decimal).sum()
        card_f_dash = np.count_nonzero(self.phi)
        # m, n = a.shape
        h, hlines = self.calcH(a, vec)
        v, vlines = self.calcH(np.rot90(a), vec)
        norm_part = [card_f + h[hlines[i, j]] + v[vlines[self.phi.shape[1] - 1 - j, i]] for i, j in product(range(self.phi.shape[0]), range(self.phi.shape[1]))]
        norm_part = np.array(norm_part).astype(Decimal).reshape(self.phi.shape)
        phi_norm = (pow(4, 4) * self.phi / pow(norm_part, 4)).sum()
        phi_sum = np.sum(self.phi)
        q1 = 0 if card_f_dash == 0 else phi_norm / card_f_dash
        if self.verbose:
            print("Compute2 {}".format(vec), self.quad1, self.quad2, self.quad3, self.quad4, self.phi, phi_sum, sep='\n')
        et = time()
        if self.logname is not None:
            if self.verbose:
                print('logging to {}'.format(self.logname))
            with open(self.logname, 'ab') as logfile:
                np.savetxt(logfile, self.phi.astype(float))
        if self.feature_image is not None:
            if self.verbose:
                print('outputting image to {}'.format(self.feature_image))
            # we use log transform for better visibility
            if np.max(self.phi) > 0:
                nnn = self.phi / np.max(self.phi)
                olog = np.log(nnn.astype(float) * 255 + 1)
                oi = olog / np.max(olog) * 255
                cv2.imwrite(self.feature_image, oi.astype(int))
            else:
                print('No output image is written, since output is empty.')
        return {"q0": phi_sum, "q1": q1, "time": et - stt}

    def interlacement(self, intensity_a: int, intensity_b: int, vec: tuple = (1, 0)):
        """
        calculates convexity of an object related to another one, marked by two different intensity values
        :param intensity_a: intensity of the first object
        :param intensity_b: intensity of the second object
        :param vec: vector of examination. plain HV-convexity has the vector of (x,y) = (col,row) = (1,0)
        :return: interlacement values
        """
        vals_forward = self.compute(intensity_a, intensity_b, vec)
        vals_inverse = self.compute(intensity_b, intensity_a, vec)
        if vals_forward['q1'] + vals_inverse['q1'] > 0:
            interlacement = {"q1_inter": 2 * vals_forward['q1'] * vals_inverse['q1'] / (vals_forward['q1'] + vals_inverse['q1'])}
        else:
            interlacement = {"q1_inter": "zerodiv"}
        for i in vals_forward:
            interlacement[i + "_forward"] = vals_forward[i]
            interlacement[i + "_inverse"] = vals_inverse[i]
        return interlacement
