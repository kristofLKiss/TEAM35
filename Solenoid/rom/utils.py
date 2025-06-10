import numpy as np

def build_interpolation_matrix(fes, mesh, points):
    """
    Given a mesh, finite element space and interpolation points create an interpolation matrix
    Args
    ----
    fes: an NGSolve finite element space
    mesh: an NGMesh for which fes was created
    points: evaluation points
    """
    interp_matrix = []
    tmp = GridFunction(fes)
    for x, y in points:
        row = []
        for i in range(fes.ndof):
            tmp.vec[:] = 0
            tmp.vec[i] = 1.0
            row.append(tmp(mesh(x, y)))
        interp_matrix.append(row)
    return np.array(interp_matrix)


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


def calc_current_vector(lookup_dict, i_vec, ri_vec, I_value, w, wstep):
    """
    Generates the current vector (I) based on active slices, current values, and lookup dictionary.

    Args:
        lookup_dict (dict): Preprocessed lookup dictionary.
        i_vec (list or array): List of turn numbers.
        ri_vec (list or array): List of inner radii (in meters).
        I_value (list or array): Current values (in amperes) for each turn.
        w (float): Coil width (in meters).
        wstep (float): Step width (in meters).

    Returns:
        np.array: Current vector (I) with the same length as coil_materials.
    """
    # init binary vect
    max_index = max(lookup_dict.values()) + 1
    current_vector = np.zeros(max_index)

    # over each turn and radius
    for turn_idx, (i, ri) in enumerate(zip(i_vec, ri_vec)):
        # search start index
        start_index = lookup_dict.get((i, round(ri, 10)))
        if start_index is None:
            raise ValueError(f"Material for T{i}R{ri:.4f} not found.")

        # Fill slices with current
        num_steps = int(w / wstep)  # Needed slices
        for step in range(num_steps):
            current_radius = round(ri + step * wstep, 10)
            idx = lookup_dict.get((i, current_radius))
            if idx is not None:
                current_vector[idx] = I_value[turn_idx]  # Add turn current to slice

    return current_vector
