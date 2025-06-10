import math
import numpy as np
from sklearn.metrics import max_error
def calculate_f1(B_vals,B_ref):
    """
    TEAM 35 accuracy metric
    :param B_vals: calculated magnetic flux density values without radius offset
    :param B_ref: reference values of magnetic flux density [Bx;By]
    :return: max norm of error as defined in TEAM 35 f1
    """
    f1 = max_error(B_vals,B_ref)
    return f1
def calculate_f2(B_ref,B_vals_plus,B_vals_minus):
    """
    TEAM 35 f2 robustness metric
    :param B_ref: reference values of magnetic flux density [Bx;By]
    :param B_vals_plus: magnetic flux density values with positive offset of winding [Bx;By]
    :param B_vals_minus: magnetic flux density values with negative offset of winding [Bx;By]
    :return: max norm of error as defined in TEAM 35 f2
    """
    f2 = max(max_error(B_vals_plus,B_ref),max_error(B_vals_minus,B_ref))
    return f2
def calculate_f3(radius,w,h,sigma_coil,I):
    """
    TEAM 35 efficiency metric
    :param radius: radius of each turn
    :param w: width of turns
    :param h: height of turns
    :param sigma_coil: conductivity of turns
    :param I: current of turns
    :return: sum of Joule-losses
    """
    f3 = 0
    for i in range(len(radius)):
        f3 += (2 * math.pi * (radius[i] + w / 2) / (sigma_coil * w * h)) * I[i] ** 2
    return f3

def round_to_step(x, step):
    return np.round(x / step) * step