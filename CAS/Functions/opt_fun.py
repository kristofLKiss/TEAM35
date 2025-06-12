from sklearn.metrics import max_error
import numpy as np
import math
from ngsolve import *

def compute_Jphi(I,w,h):
    '''
    Áramsűrűség számítása áramból és geometriából
    '''
    return I / (w * h)


def generate_coordinates(x_min,x_max,y_min,y_max,x_points,y_points):
    """
    Generate coordinates in the specified format as tuples for a rectangular region.

    Parameters:
    x_min, x_max: float
        The minimum and maximum values of the x-coordinate.
    y_min, y_max: float
        The minimum and maximum values of the y-coordinate.
    x_points, y_points: int
        Number of evaluation points along x and y axes, respectively.

    Returns:
    list
        A list of tuples, each containing an (x, y) coordinate.
    """
    x_coords = np.linspace(x_min,x_max,x_points)
    y_coords = np.linspace(y_min,y_max,y_points)
    points = [(x,y) for y in y_coords for x in x_coords]
    return points


def calculate_metrics(B_vals,B_ref,B_vals_plus,B_vals_minus,n_turns,radius,w,h,sigma_coil,I):
    f1 = max_error(B_vals,B_ref)
    # print("f1 célfüggvény értéke: {:.2e}".format(f1))

    # f2 metrika: robosztusság
    f2 = max(max_error(B_vals_plus,B_ref),max_error(B_vals_minus,B_ref))
    # print("f2 célfüggvény értéke: {:.2e}".format(f2))

    # f3 metrika: Joule-veszteség
    f3 = 0
    for i in range(n_turns):
        f3 += (2 * math.pi * (radius[i] + w / 2) / (sigma_coil * w * h)) * I[i] ** 2
    # print("f3 célfüggvény értéke:",f3)

    return f1,f2,f3

import pickle

def load_ngsolve_data(file_path):

    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Hiba: A fájl '{file_path}' nem található.")
    except Exception as e:
        print(f"Hiba történt a betöltés során: {e}")



def calculate_B_fields(FEspace, rAphi, r):
    """
    Calculate the magnetic field components (Br and Bz) in a cylindrical coordinate system.

    Args:
        FEspace: Finite element space (NGSolve H1 space) where the solution is defined.
        rAphi: GridFunction defined in the FEspace, representing r * A (vector potential in cylindrical coordinates).
        r: CoefficientFunction representing the radial coordinate (e.g., r = x + epsilon), avoiding division by zero.

    Returns:
        Tuple of GridFunctions:
            - Br_sol: GridFunction containing the radial component of the magnetic field (Br).
            - Bz_sol: GridFunction containing the axial component of the magnetic field (Bz).
    """
    Aphi = GridFunction(FEspace)
    Aphi.Set(rAphi / r)
    Br_sol = GridFunction(FEspace)
    Bz_sol = GridFunction(FEspace)
    B_r = -grad(Aphi)[1]
    B_z = (1 / r) * grad(rAphi)[0]
    Br_sol.Set(B_r)
    Bz_sol.Set(B_z)
    return Br_sol, Bz_sol

def preprocess_coil_materials(coil_materials):
    """
    Prepares a lookup dictionary for efficient index-based searches.

    Args:
        coil_materials (list): List of material names in the format 'T{i}R{ri}'.

    Returns:
        dict: Lookup dictionary with keys as (i, ri) tuples and values as list indices.
    """
    lookup_dict = {}
    for idx, material in enumerate(coil_materials):
        if material.startswith("T"):
            parts = material[1:].split("R")
            i = int(parts[0])  # Extract the turn number
            ri = float(parts[1])  # Extract the radius
            lookup_dict[(i, ri)] = idx
    return lookup_dict

def calc_current_vector(lookup_dict, n_turns, i_vec, ri_vec, I_value, w, wstep):
    """
    Generates the current vector (I) based on active slices, current values, and lookup dictionary.

    Args:
        lookup_dict (dict): Preprocessed lookup dictionary.
        n_turns (int): Number of turns in the coil.
        i_vec (list or array): List of turn numbers.
        ri_vec (list or array): List of inner radii (in meters).
        I_value (list or array): Current values (in amperes) for each turn.
        w (float): Coil width (in meters).
        wstep (float): Step width (in meters).

    Returns:
        np.array: Current vector (I) with the same length as coil_materials.
    """
    # Bináris vektor inicializálása
    max_index = max(lookup_dict.values()) + 1
    current_vector = np.zeros(max_index)

    # Iterálás az összes menet és sugár felett
    for turn_idx, (i, ri) in enumerate(zip(i_vec, ri_vec)):
        # Kezdő index megkeresése
        start_index = lookup_dict.get((i, round(ri, 10)))
        if start_index is None:
            raise ValueError(f"Material for T{i}R{ri:.4f} not found.")

        # Szeletek aktuális menet áramával való kitöltése
        num_steps = int(w / wstep)  # Szükséges szeletek száma
        for step in range(num_steps):
            current_radius = round(ri + step * wstep, 10)
            idx = lookup_dict.get((i, current_radius))
            if idx is not None:
                current_vector[idx] = I_value[turn_idx]  # Menet áramát hozzárendeljük

    return current_vector


def calculate_errors(Inp1,Inp2):
    """
    """
    avg_abs_error = np.mean(np.abs(Inp1-Inp2))
    max_abs_error = np.max(np.abs(Inp1-Inp2))
    avg_rel_error = np.mean(np.abs(Inp1-Inp2) / Inp2)
    max_rel_error = np.max(np.abs(Inp1-Inp2) / Inp2)

    return avg_abs_error,max_abs_error,avg_rel_error,max_rel_error