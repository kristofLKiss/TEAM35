import time
from .generateQuadMesh import generate_ngsolve_geometry_and_mesh_w_ins_offset_geomopt, generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane
from .opt_fun import *
import pickle
import numpy as np
from math import sqrt
from ngsolve import *
import matplotlib.pyplot as plt

import time
import math
import numpy as np
import pickle
from ngsolve import *

def generate_reduced_order_model_sourceopt(n_turns, w, h, insulation_thickness, solh, solw, radius,
                                               maxh, controlledmaxh, save_filename):
    """
    Generate a reduced-order model for a magnetostatic field with offset geometries.

    Parameters:
    - n_turns: Number of turns in the coil
    - w: Width of the coil
    - h: Height of the coil
    - insulation_thickness: Thickness of the insulation
    - solh: Height of the control region
    - solw: Width of the control region
    - radius: Radius of the coil
    - maxh: Maximum mesh element size
    - controlledmaxh: Controlled mesh element size
    - coil_materials: List of coil materials
    - save_filename: Name of the file to save the model data

    Returns:
    - Dictionary containing reduced-order model data and meshes
    """
    timings = {}
    start_time = time.process_time()

    # Generate meshes for the main, plus-offset, and minus-offset geometries
    mesh, geo = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, 0, maxh, controlledmaxh, 0)
    mesh_plus, geo_plus = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, 0.5e-3, maxh, controlledmaxh, 0)
    mesh_minus, geo_minus = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, -0.5e-3, maxh, controlledmaxh, 0)

    meshing_time = time.process_time()
    print(f"Meshes generated in {meshing_time - start_time} seconds")
    timings["meshing"] = meshing_time - start_time

    nu0 = CoefficientFunction(1e7 / (4 * math.pi))
    coil_materials = [f"T{i}" for i in range(1,11)]

    # Define finite element spaces
    Vh = H1(mesh, order=2, dirichlet="left|right|top")
    Vh_plus = H1(mesh_plus, order=2, dirichlet="left|right|top")
    Vh_minus = H1(mesh_minus, order=2, dirichlet="left|right|top")
    Vh_red = H1(mesh, order=2, definedon="Control")
    Vh_plus_red = H1(mesh_plus, order=2, definedon="Control")
    Vh_minus_red = H1(mesh_minus, order=2, definedon="Control")
    # Trial and test functions
    Aphi = Vh.TrialFunction()
    v = Vh.TestFunction()
    Aphi_plus = Vh_plus.TrialFunction()
    v_plus = Vh_plus.TestFunction()
    Aphi_minus = Vh_minus.TrialFunction()
    v_minus = Vh_minus.TestFunction()

    # Bilinear forms
    r = x + 1e-8  # Avoid division by zero
    a = BilinearForm(Vh)
    a += nu0 * grad(Aphi) * grad(v) * (1 / r) * dx
    a.Assemble()
    a_plus = BilinearForm(Vh_plus)
    a_plus += nu0 * grad(Aphi_plus) * grad(v_plus) * (1 / r) * dx
    a_plus.Assemble()
    a_minus = BilinearForm(Vh_minus)
    a_minus += nu0 * grad(Aphi_minus) * grad(v_minus) * (1 / r) * dx
    a_minus.Assemble()


    RomAred = []
    RomAredPlus = []
    RomAredMinus = []
    # Loop over each coil material to compute reduced-order solutions
    for i, material in enumerate(coil_materials):
        print(f"Index: {i}, Material: {material}")

        # Define linear forms for each geometry
        f = LinearForm(Vh)
        f += Parameter(compute_Jphi(1, w, h)) * v * dx(material)
        f.Assemble()
        f_plus = LinearForm(Vh_plus)
        f_plus += Parameter(compute_Jphi(1, w, h)) * v_plus * dx(material)
        f_plus.Assemble()
        f_minus = LinearForm(Vh_minus)
        f_minus += Parameter(compute_Jphi(1, w, h)) * v_minus * dx(material)
        f_minus.Assemble()

        # Solve systems
        Aphi_sol = GridFunction(Vh)
        Aphi_sol_red = GridFunction(Vh_red)
        Aphi_sol.vec.data = a.mat.Inverse(Vh.FreeDofs()) * f.vec
        Aphi_sol_red.Set(Aphi_sol)

        Aphi_sol_plus = GridFunction(Vh_plus)
        Aphi_sol_plus.vec.data = a_plus.mat.Inverse(Vh_plus.FreeDofs()) * f_plus.vec
        Aphi_sol_red_plus = GridFunction(Vh_plus_red)
        Aphi_sol_red_plus.Set(Aphi_sol_plus)

        Aphi_sol_minus = GridFunction(Vh_minus)
        Aphi_sol_minus.vec.data = a_minus.mat.Inverse(Vh_minus.FreeDofs()) * f_minus.vec
        Aphi_sol_red_minus = GridFunction(Vh_minus_red)
        Aphi_sol_red_minus.Set(Aphi_sol_minus)
        # Append reduced solutions

        RomAred.append(Aphi_sol_red.vec.data)
        RomAredPlus.append(Aphi_sol_red_plus.vec.data)
        RomAredMinus.append(Aphi_sol_red_minus.vec.data)

    RomAred = np.array(RomAred)
    RomAredPlus = np.array(RomAredPlus)
    RomAredMinus = np.array(RomAredMinus)
    generation_time = time.process_time()
    print(f"Reduced order models generated in {generation_time - meshing_time} seconds")
    timings["generation"] = generation_time - meshing_time

    # Save data to file
    data = {
        "RomAred": RomAred,
        "RomAredPlus": RomAredPlus,
        "RomAredMinus": RomAredMinus,
        "FEspace_red": Vh_red,
        "FEspace_redPlus": Vh_plus_red,
        "FEspace_redMinus": Vh_minus_red,
        "mesh" : mesh,
        "mesh_plus": mesh_plus,
        "mesh_minus": mesh_minus
    }

    with open(save_filename, "wb") as file:
        pickle.dump(data, file)

    save_time = time.process_time()
    print(f"Data saved in {save_time - generation_time} seconds")
    timings["saving"] = save_time - generation_time

    return data, mesh, mesh_minus, mesh_plus

def generate_reduced_order_model_geomopt(n_turns, w, h, insulation_thickness, solh, solw, maxh, controlledmaxh,
                                  wstep, minradius, maxradius, save_filename):
    """
    Generate a reduced-order model for a magnetostatic field for TEAM35 geometry optimization problem.

    Parameters:
    - n_turns: Number of turns in the coil
    - w: Width of the coil
    - h: Height of the coil
    - insulation_thickness: Thickness of the insulation
    - solh: Height of the control region
    - solw: Width of the control region
    - maxh: Maximum mesh element size
    - controlledmaxh: Controlled mesh element size
    - wstep: Width step for geometry generation
    - minradius: Minimum radius for the coil
    - maxradius: Maximum radius for the coil
    - save_filename: Name of the file to save the model data

    Returns:
    - Dictionary containing reduced order model data and mesh
    """
    timings = {}
    # Start timer
    start_time = time.process_time()

    # Generate mesh and geometry
    mesh, geo = generate_ngsolve_geometry_and_mesh_w_ins_offset_geomopt(
        n_turns, wstep, h, insulation_thickness, solh, solw, minradius, maxradius, maxh, controlledmaxh, 0
    )

    meshing_time = time.process_time()
    print(f"Mesh generated in {meshing_time - start_time} seconds")
    timings["meshing"] = meshing_time - start_time
    materials = mesh.GetMaterials()
    coil_materials = [material for material in materials if material.startswith("T")]
    nu0 = 1e7 / (4 * math.pi)

    # Create H1 finite element space
    Vh = H1(mesh, order=2, dirichlet="left|right|top")
    f1 = H1(mesh, order=2, definedon="Control")

    # Define trial and test functions
    rAphi_tf = Vh.TrialFunction()
    v = Vh.TestFunction()

    # Define bilinear form
    r = x + 1e-8  # Avoid division by zero
    a = BilinearForm(Vh)
    a += nu0 * grad(rAphi_tf) * grad(v) * (1 / r) * dx
    a.Assemble()
    aInv = a.mat.Inverse(Vh.FreeDofs())

    rA = []
    rAred = []

    # Current parameters
    I = 1
    J_phi = Parameter(compute_Jphi(I, w, h))

    # Loop over coil materials
    for i, material in enumerate(coil_materials):
        # print(f"Index: {i}, Material: {material}")

        # Define linear form
        f = LinearForm(Vh)
        f += J_phi * v * dx(material)
        f.Assemble()

        # Solve system
        rAphi_sol = GridFunction(Vh)
        rAphi_sol.vec.data = aInv * f.vec

        # Reduced solution
        rAphiRed = GridFunction(f1)
        rAphiRed.Set(rAphi_sol)

        # Store results
        rA.append(rAphi_sol.vec.data)
        rAred.append(rAphiRed.vec.data)

        # print(f"Calculation done: {i + 1}/{len(coil_materials)}")

    generation_time = time.process_time()
    print(f"Reduced order model generated in {generation_time - meshing_time} seconds")
    timings["generation"] = generation_time - meshing_time
    rA = np.array(rA)
    rAred = np.array(rAred)

    # # Save data to file
    # data = {
    #     "RomA": rA,
    #     "RomAred": rAred,
    #     "mesh": mesh,
    #     "FEspace": Vh,
    #     "FEspace_red": f1
    # }
    # with open(save_filename, "wb") as file:
    #     pickle.dump(data, file)
    #
    save_time = time.process_time()
    materials = mesh.GetMaterials()
    coil_materials = [material for material in materials if material.startswith("T")]
    CM = preprocess_coil_materials(coil_materials)
    print(f"Data saved in {save_time - generation_time} seconds")
    timings["saving"] = save_time - meshing_time
    return mesh, f1, rAred, CM

def load_radopt(filename,n_turns,i_vec,I,radius):
    """
    Load data from a pickle file and preprocess the materials.

    Args:
        filename: Path to the pickle file containing reduced-order model data.
        n_turns: Number of turns in the coil.
        i_vec: Current vector.
        I: Current amplitude array.
        radius: Radial positions of the coils.

    Returns:
        Tuple containing:
            - mesh: Mesh object from the loaded data.
            - Vh: Finite element space from the loaded data.
            - rA: Reduced-order model data for the magnetic vector potential.
            - CM: Preprocessed coil materials.
    """
    with open(filename, "rb") as file:
        data = pickle.load(file)

    mesh = data["mesh"]
    # Vh = data["FEspace"]
    # rA = np.array(data["RomA"])

    # Optional: Load reduced FE space and rAred
    Vh = data["FEspace_red"]
    rA = np.array(data["RomAred"])

    # Ensure inputs have the correct length
    if not (len(radius) == len(i_vec) == len(I) == n_turns):
        raise ValueError("Length of radius, i_vec, and I must match n_turns.")

    materials = mesh.GetMaterials()
    coil_materials = [material for material in materials if material.startswith("T")]
    CM = preprocess_coil_materials(coil_materials)

    return mesh, Vh, rA, CM

def load_sourceopt(filename):
    """
    Load data from a pickle file.

    Args:
        filename: Path to the pickle file containing reduced-order model data.

    Returns:
        Tuple containing:
            - RomAred: Reduced-order model data.
            - RomAredPlus: Reduced-order model data for positive offset geometry.
            - RomAredMinus: Reduced-order model data for negative offset geometry.
            - FEspace_red: Finite element space for the control region.
            - FEspace_redPlus: Finite element space for the control region.
            - FEspace_redMinus: Finite element space for the control region.
    """
    with open(filename, "rb") as file:
        data = pickle.load(file)

    RomAred = data.get("RomAred")
    RomAredPlus = data.get("RomAredPlus")
    RomAredMinus = data.get("RomAredMinus")
    FEspace_red = data.get("FEspace_red")
    FEspace_redPlus = data.get("FEspace_redPlus")
    FEspace_redMinus = data.get("FEspace_redMinus")
    mesh = data.get("mesh")
    mesh_plus = data.get("mesh_plus")
    mesh_minus =data.get("mesh_minus")

    return RomAred, RomAredPlus, RomAredMinus, FEspace_red, FEspace_redPlus, FEspace_redMinus, mesh, mesh_minus, mesh_plus


def calc_with_ROM(mesh, Vh, rA, CM, n_turns, i_vec, I, radius, coordinates, B_ref, w, wstep, h, sigma_coil):
    """
    Perform postprocessing steps to calculate magnetic field components.

    Args:
        mesh: Mesh object from the loaded data.
        Vh: Finite element space from the loaded data.
        rA: Reduced-order model data for the magnetic vector potential.
        CM: Preprocessed coil materials.
        n_turns: Number of turns in the coil.
        i_vec: Current vector.
        I: Current amplitude array.
        radius: Radial positions of the coils.
        coordinates: List of coordinate points to evaluate the field.
        B_ref: Reference magnetic flux density values.
        w: Width of the coil.
        wstep: Width step for geometry generation.
        h: Height of the coil.
        sigma_coil: Electrical conductivity of the coil.

    Returns:
        Metrics calculated during the postprocessing.
    """
    timings = {}
    start_time = time.process_time()

    # Initialize GridFunctions
    rA_surr = GridFunction(Vh)
    rA_surr_plus = GridFunction(Vh)
    rA_surr_minus = GridFunction(Vh)

    f = calc_current_vector(CM, n_turns, i_vec, radius, I, w, wstep)
    # print(f)
    fplus = calc_current_vector(CM, n_turns, i_vec, np.array(radius) + 0.5e-3, I, w, wstep)
    # print(fplus)
    fminus = calc_current_vector(CM, n_turns, i_vec, np.array(radius) - 0.5e-3, I, w, wstep)
    # print(fminus)

    rA_surr.vec.data = np.dot(f, rA)
    rA_surr_plus.vec.data = np.dot(fplus, rA)
    rA_surr_minus.vec.data = np.dot(fminus, rA)
    vectorpot_time = time.process_time()
    timings["vectorpot"] = vectorpot_time - start_time
    # print(f"solved for vector potential in {timings['vectorpot']:.6f} seconds")
    # Calculate magnetic fields using helper function
    r = CoefficientFunction(x + 1e-8)  # Avoid division by zero
    Br_sol, Bz_sol = calculate_B_fields(Vh, rA_surr, r)
    Br_sol_plus, Bz_sol_plus = calculate_B_fields(Vh, rA_surr_plus, r)
    Br_sol_minus, Bz_sol_minus = calculate_B_fields(Vh, rA_surr_minus, r)

    # Evaluate field values at given coordinates
    Br_vals = [Br_sol(mesh(*p)) for p in coordinates]
    Bz_vals = [Bz_sol(mesh(*p)) for p in coordinates]
    Br_vals_plus = [Br_sol_plus(mesh(*p)) for p in coordinates]
    Bz_vals_plus = [Bz_sol_plus(mesh(*p)) for p in coordinates]
    Br_vals_minus = [Br_sol_minus(mesh(*p)) for p in coordinates]
    Bz_vals_minus = [Bz_sol_minus(mesh(*p)) for p in coordinates]

    # Calculate total magnetic field
    B_vals = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals, Bz_vals)])
    B_vals_plus = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_plus, Bz_vals_plus)])
    B_vals_minus = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_minus, Bz_vals_minus)])

    field_computation_time = time.process_time()
    timings["field_computation"] = field_computation_time - vectorpot_time
    # print(f"computed magnetic fields in {timings['field_computation']:.6f} seconds")

    # Calculate metrics
    [f1,f2,f3] = calculate_metrics(B_vals, B_ref, B_vals_plus, B_vals_minus, n_turns, radius, w, h, sigma_coil, I)
    metric_computation_time = time.process_time()
    timings["metric_computation"] = metric_computation_time - start_time
    # print(f"computed generation in {timings['metric_computation']:.6f} seconds")

    return Br_vals, Bz_vals, B_vals, f1, f2, f3, timings, rA_surr, Br_sol,Bz_sol

def calc_with_ROM_source_optimization(mesh, mesh_plus, mesh_minus, RomAred, RomAredPlus, RomAredMinus, FEspace_red, FEspace_redPlus, FEspace_redMinus, I, radius=10e-3, coordinates=None, B_ref=None, n_turns=10,w=0.0010,h=0.0015,sigma_coil=4e7):
    """
    Perform source optimization calculations using reduced-order models.

    Args:
        mesh: Mesh object from the loaded data.
        mesh_plus: Mesh object for positive offset.
        mesh_minus: Mesh object for positive offset.
        RomAred: Reduced-order model data for the magnetic vector potential.
        RomAredPlus: Reduced-order model data for the positive offset geometry.
        RomAredMinus: Reduced-order model data for the negative offset geometry.
        FEspace_red: Finite element space for the base geometry.
        FEspace_redPlus: Finite element space for the positive offset geometry.
        FEspace_redMinus: Finite element space for the negative offset geometry.
        I: Current amplitude array (10 elements).
        radius: Radial position of the coils (default 10 mm).
        coordinates: List of coordinate points to evaluate the field.
        B_ref: Reference magnetic flux density values.

    Returns:
        Metrics calculated during the postprocessing.
    """
    timings = {}
    start_time = time.process_time()

    # Initialize GridFunctions
    rA_surr = GridFunction(FEspace_red)
    rA_surr_plus = GridFunction(FEspace_redPlus)
    rA_surr_minus = GridFunction(FEspace_redMinus)

    # Calculate surrogate vector potentials using the reduced-order models
    rA_surr.vec.data = np.dot(I, RomAred)
    # print(rA_surr.vec.data)
    rA_surr_plus.vec.data = np.dot(I, RomAredPlus)
    # print(rA_surr_plus.vec.data)
    rA_surr_minus.vec.data = np.dot(I, RomAredMinus)

    vectorpot_time = time.process_time()
    timings["vectorpot"] = vectorpot_time - start_time

    # Calculate magnetic fields using helper function
    r = CoefficientFunction(x + 1e-8)  # Avoid division by zero
    Br_sol, Bz_sol = calculate_B_fields(FEspace_red, rA_surr, r)
    Br_sol_plus, Bz_sol_plus = calculate_B_fields(FEspace_redPlus, rA_surr_plus, r)
    Br_sol_minus, Bz_sol_minus = calculate_B_fields(FEspace_redMinus, rA_surr_minus, r)

    # Evaluate field values at given coordinates
    if coordinates is not None:
        Br_vals = [Br_sol(mesh(*p)) for p in coordinates]
        Bz_vals = [Bz_sol(mesh(*p)) for p in coordinates]
        Br_vals_plus = [Br_sol_plus(mesh_plus(*p)) for p in coordinates]
        Bz_vals_plus = [Bz_sol_plus(mesh_plus(*p)) for p in coordinates]
        Br_vals_minus = [Br_sol_minus(mesh_minus(*p)) for p in coordinates]
        Bz_vals_minus = [Bz_sol_minus(mesh_minus(*p)) for p in coordinates]

        # Calculate total magnetic field
        B_vals = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals, Bz_vals)])
        B_vals_plus = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_plus, Bz_vals_plus)])
        B_vals_minus = np.array([sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_minus, Bz_vals_minus)])

        field_computation_time = time.process_time()
        timings["field_computation"] = field_computation_time - vectorpot_time

        # Calculate metrics
        f1, f2, f3 = calculate_metrics(B_vals,B_ref,B_vals_plus,B_vals_minus,n_turns,[radius] * 10,w,h,sigma_coil,I)
        metric_computation_time = time.process_time()
        timings["metric_computation"] = metric_computation_time - start_time

        return Br_vals, Bz_vals, B_vals, f1, f2, f3, timings, rA_surr, Br_sol, Bz_sol