"""Low-level geometry and coordinate utilities shared by the detection pipeline.

matrix2xyz       Matrix -> list of (row, col, value) triples.
fit_ellipse      Algebraic ellipse fit from contour points (Fitzgibbon 1999).
ellipse_center   Centre of a fitted conic section.
ellipse_angle_of_rotation  Principal axis angle in radians (0 = East).
ellipse_axis_length        Semi-axes from fitted conic; [nan, nan] for a
                           degenerate conic (callers drop such masks).
cart2pol / pol2cart         Cartesian <-> polar transforms (array-safe).
"""
import numpy as np
from numpy.linalg import inv, svd
from math import atan2

def matrix2xyz(matrix):
    m = np.tile(range(0,np.shape(matrix)[1]),[np.shape(matrix)[0],1])
    n = np.transpose(np.tile(range(0,np.shape(matrix)[0]),[np.shape(matrix)[1],1]))
    xyz = np.transpose([n[matrix==matrix],m[matrix==matrix],matrix[matrix==matrix]])
    return (xyz);
    
def fit_ellipse(x,y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return (a)

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return (np.array([x0, y0]))

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    phi = atan2(2 * b, (a - c)) / 2
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return (phi)

def ellipse_axis_length( a ):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    # A degenerate conic (collinear or too few contour points) returns NaN
    # explicitly; _measure_clast drops rows whose Clast_length is NaN.
    denom_disc = (a - c) * (a - c)
    if denom_disc == 0:
        return np.array([np.nan, np.nan])
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / denom_disc) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / denom_disc) - (c + a)
    )
    if down1 == 0 or down2 == 0:
        return np.array([np.nan, np.nan])
    r1_sq = up / down1
    r2_sq = up / down2
    if r1_sq < 0 or r2_sq < 0 or not np.isfinite(r1_sq) or not np.isfinite(r2_sq):
        return np.array([np.nan, np.nan])
    res1 = np.sqrt(r1_sq)
    res2 = np.sqrt(r2_sq)
    return (np.array([np.max([res1, res2]), np.min([res1, res2])]))

def cart2pol(x, y):
    # np.arctan2 returns [-pi, pi]; map the negative half onto (pi, 2pi).
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
