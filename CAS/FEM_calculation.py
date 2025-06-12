from ngsolve import *
from Functions.generateQuadMesh import generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane_radopt,generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane
from Functions.opt_fun import *
import math
import time

def fem_simulation(n_turns=10, w=0.001, h=0.0015, radius=None,
                    insulation_thickness=0.00006, solh=0.005, solw=0.005, maxh=0.0002,
                    controlledmaxh=0.0002, B_target=2e-3, I=None, sigma_coil=4e7):
    """
    Perform FEM simulation to calculate magnetic field components and metrics of the TEAM35 optimization problem.

    Args:
        n_turns: Number of coil turns.
        w: Width of the coil.
        h: Height of the coil.
        radius: List of radii for the coils.
        gap: Gap between coils.
        current_density: Current density in the coils.
        insulation_thickness: Thickness of the insulation layer.
        solh: Height of the solution domain.
        solw: Width of the solution domain.
        maxh: Maximum mesh element size.
        controlledmaxh: Controlled maximum mesh element size.
        B_target: Target magnetic field strength.
        I: Current values in each coil.
        sigma_coil: Electrical conductivity of the coils.

    Returns:
        Tuple:
            - Br_vals: List of radial magnetic field values.
            - Bz_vals: List of axial magnetic field values.
            - B_vals: List of total magnetic field magnitudes.
            - f1, f2, f3: Metrics calculated based on the simulation.
            - timings: Dictionary of measured times for different steps.
    """
    # Default radius and current values if not provided
    if radius is None:
        radius = [r * 1e-3 for r in [10,10,10,10,10,10,10,10,10,10]]
    if I is None:
        I = np.ones(10) * 3.18

    timings = {}

    # Magnetic reluctivity coefficient (nu0)
    nu0 = CoefficientFunction(1 / (4 * math.pi * 1e-7))

    # Generate evaluation points within the domain
    coordinates = generate_coordinates(0.0002, solw, 0, solh, 11, 6)
    B_ref = [B_target for _ in coordinates]  # Reference magnetic field values

    # Mesh generation for three offset configurations (nominal, positive, negative 0.5 mm)
    start_time = time.process_time()

    # Directly assign meshes to variables without intermediate list
    mesh, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane_radopt(
        n_turns=n_turns, w=w, h=h, insulation_thickness=insulation_thickness, solh=solh, solw=solw, radius=radius, dr=0, maxh=maxh, controlledmaxh=controlledmaxh, visu=0
    )
    mesh_plus, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane_radopt(
        n_turns=n_turns, w=w, h=h, insulation_thickness=insulation_thickness, solh=solh, solw=solw, radius=radius, dr=0.5e-3, maxh=maxh, controlledmaxh=controlledmaxh, visu=0
    )
    mesh_minus, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane_radopt(
        n_turns=n_turns, w=w, h=h, insulation_thickness=insulation_thickness, solh=solh, solw=solw, radius=radius, dr=-0.5e-3, maxh=maxh, controlledmaxh=controlledmaxh, visu=0
    )

    meshing_time = time.process_time()
    timings["meshing"] = meshing_time - start_time
    print(f"generated meshes in {timings['meshing']:.6f} seconds")

    # Define finite element spaces for the meshes
    Vh = H1(mesh, order=2, dirichlet="left|right|top")
    Vhplus = H1(mesh_plus, order=2, dirichlet="left|right|top")
    Vhminus = H1(mesh_minus, order=2, dirichlet="left|right|top")

    # Define trial and test functions for each space
    Aphi, v = Vh.TrialFunction(), Vh.TestFunction()
    Aphiplus, vplus = Vhplus.TrialFunction(), Vhplus.TestFunction()
    Aphiminus, vminus = Vhminus.TrialFunction(), Vhminus.TestFunction()

    # Define material labels and compute current densities
    coil_materials = [f"T{i}" for i in range(1, 11)]
    J_phi = [Parameter(compute_Jphi(I[i], w, h)) for i in range(len(coil_materials))]

    # Radial coordinate (avoiding division by zero)
    r = x + 1e-8

    # Bilinear forms (stiffness matrices) for each configuration
    a = BilinearForm(Vh)
    a += nu0 * grad(Aphi) * grad(v) * (1 / r) * dx
    aplus = BilinearForm(Vhplus)
    aplus += nu0 * grad(Aphiplus) * grad(vplus) * (1 / r) * dx
    aminus = BilinearForm(Vhminus)
    aminus += nu0 * grad(Aphiminus) * grad(vminus) * (1 / r) * dx

    # Linear forms (source terms) for each configuration
    f = LinearForm(Vh)
    fplus = LinearForm(Vhplus)
    fminus = LinearForm(Vhminus)
    for i, material in enumerate(coil_materials):
        f += J_phi[i] * v * dx(material)
        fplus += J_phi[i] * v * dx(material)
        fminus += J_phi[i] * v * dx(material)

    # Assemble bilinear and linear forms
    f.Assemble()
    fplus.Assemble()
    fminus.Assemble()
    a.Assemble()
    aplus.Assemble()
    aminus.Assemble()

    # Solve the systems for the three configurations
    start_time = time.process_time()
    rAphi_sol, rAphi_sol_plus, rAphi_sol_minus = GridFunction(Vh), GridFunction(Vhplus), GridFunction(Vhminus)
    rAphi_sol.vec.data = a.mat.Inverse(Vh.FreeDofs()) * f.vec
    rAphi_sol_plus.vec.data = aplus.mat.Inverse(Vhplus.FreeDofs()) * fplus.vec
    rAphi_sol_minus.vec.data = aminus.mat.Inverse(Vhminus.FreeDofs()) * fminus.vec

    solution_time = time.process_time()
    timings["solution"] = solution_time - start_time
    print(f"solved FEM models in {timings['solution']:.6f} seconds")

    # Reduced FE spaces for post-processing
    start_time = time.process_time()
    Vh_red = H1(mesh, definedon="Control")
    Vh_red_plus = H1(mesh_plus, definedon="Control")
    Vh_red_minus = H1(mesh_minus, definedon="Control")

    # Compute magnetic field components (Br, Bz) for each configuration
    Br_sol, Bz_sol = calculate_B_fields(Vh_red, rAphi_sol, r)
    Br_sol_plus, Bz_sol_plus = calculate_B_fields(Vh_red_plus, rAphi_sol_plus, r)
    Br_sol_minus, Bz_sol_minus = calculate_B_fields(Vh_red_minus, rAphi_sol_minus, r)



    # Evaluate field values at predefined points
    Br_vals = [Br_sol(mesh(*p)) for p in coordinates]
    Bz_vals = [Bz_sol(mesh(*p)) for p in coordinates]
    Br_vals_plus = [Br_sol_plus(mesh_plus(*p)) for p in coordinates]
    Bz_vals_plus = [Bz_sol_plus(mesh_plus(*p)) for p in coordinates]
    Br_vals_minus = [Br_sol_minus(mesh_minus(*p)) for p in coordinates]
    Bz_vals_minus = [Bz_sol_minus(mesh_minus(*p)) for p in coordinates]

    # Calculate total magnetic field magnitude
    B_vals = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals, Bz_vals)])
    B_vals_plus = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_plus, Bz_vals_plus)])
    B_vals_minus = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_minus, Bz_vals_minus)])

    field_computation_time = time.process_time()
    timings["field_computation"] = field_computation_time - start_time
    print(f"computed magnetic fields in {timings['field_computation']:.6f} seconds")

    # Compute metrics based on the simulation
    start_time = time.process_time()
    [f1, f2, f3] = calculate_metrics(B_vals, B_ref, B_vals_plus, B_vals_minus, n_turns, radius, w, h, sigma_coil, I)
    metric_computation_time = time.process_time()
    timings["metric_computation"] = metric_computation_time - start_time
    print(f"computed metrics in {timings['metric_computation']:.6f} seconds")

    return Br_vals, Bz_vals, B_vals, f1, f2, f3, timings, mesh, rAphi_sol, Br_sol, Bz_sol
#
# [Br_vals, Bz_vals, B_vals, f1, f2, f3, timings] =  fem_simulation(n_turns=10, w=0.001, h=0.0015, radius=None,
#                     insulation_thickness=0.00006, solh=0.005, solw=0.005, maxh=0.0001,
#                     controlledmaxh=0.0001, B_target=2e-3, I=None, sigma_coil=4e7)

def fem_simulation_sourceopt(n_turns=10, w=0.001, h=0.0015, radius=None,
                    insulation_thickness=0.00006, solh=0.005, solw=0.005, maxh=0.0001,
                    controlledmaxh=0.0001, B_target=2e-3, I=None, sigma_coil=4e7):
    """
    Perform FEM simulation to calculate magnetic field components and metrics of the TEAM35 optimization problem.

    Args:
        n_turns: Number of coil turns.
        w: Width of the coil.
        h: Height of the coil.
        radius: List of radii for the coils.
        gap: Gap between coils.
        current_density: Current density in the coils.
        insulation_thickness: Thickness of the insulation layer.
        solh: Height of the solution domain.
        solw: Width of the solution domain.
        maxh: Maximum mesh element size.
        controlledmaxh: Controlled maximum mesh element size.
        B_target: Target magnetic field strength.
        I: Current values in each coil.
        sigma_coil: Electrical conductivity of the coils.

    Returns:
        Tuple:
            - Br_vals: List of radial magnetic field values.
            - Bz_vals: List of axial magnetic field values.
            - B_vals: List of total magnetic field magnitudes.
            - f1, f2, f3: Metrics calculated based on the simulation.
            - timings: Dictionary of measured times for different steps.
    """
    # Default radius and current values if not provided
    if radius is None:
        radius = 10e-3
    if I is None:
        I = np.ones(10) * 3.18

    timings = {}

    # Magnetic reluctivity coefficient (nu0)
    nu0 = CoefficientFunction(1 / (4 * math.pi * 1e-7))

    # Generate evaluation points within the domain
    coordinates = generate_coordinates(0.0002, solw, 0, solh, 11, 6)
    B_ref = [B_target for _ in coordinates]  # Reference magnetic field values

    # Mesh generation for three offset configurations (nominal, positive, negative 0.5 mm)
    print("Start meshing")
    start_time = time.process_time()


    # Directly assign meshes to variables without intermediate list
    mesh, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, 0, maxh, controlledmaxh, 0)
    mesh_plus, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, 0.5e-3, maxh, controlledmaxh, 0)
    mesh_minus, _ = generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns, w, h, insulation_thickness, solh, solw, radius, -0.5e-3, maxh, controlledmaxh, 0)

    meshing_time = time.process_time()
    timings["meshing"] = meshing_time - start_time
    print(f"generated meshes in {timings['meshing']:.6f} seconds")

    # Define finite element spaces for the meshes
    Vh = H1(mesh, order=2, dirichlet="left|right|top")
    Vhplus = H1(mesh_plus, order=2, dirichlet="left|right|top")
    Vhminus = H1(mesh_minus, order=2, dirichlet="left|right|top")

    # Define trial and test functions for each space
    Aphi, v = Vh.TrialFunction(), Vh.TestFunction()
    Aphiplus, vplus = Vhplus.TrialFunction(), Vhplus.TestFunction()
    Aphiminus, vminus = Vhminus.TrialFunction(), Vhminus.TestFunction()

    # Define material labels and compute current densities
    coil_materials = [f"T{i}" for i in range(1, 11)]
    J_phi = [Parameter(compute_Jphi(I[i], w, h)) for i in range(len(coil_materials))]

    # Radial coordinate (avoiding division by zero)
    r = x + 1e-8

    # Bilinear forms (stiffness matrices) for each configuration
    a = BilinearForm(Vh)
    a += nu0 * grad(Aphi) * grad(v) * (1 / r) * dx
    aplus = BilinearForm(Vhplus)
    aplus += nu0 * grad(Aphiplus) * grad(vplus) * (1 / r) * dx
    aminus = BilinearForm(Vhminus)
    aminus += nu0 * grad(Aphiminus) * grad(vminus) * (1 / r) * dx

    # Linear forms (source terms) for each configuration
    f = LinearForm(Vh)
    fplus = LinearForm(Vhplus)
    fminus = LinearForm(Vhminus)
    for i, material in enumerate(coil_materials):
        f += J_phi[i] * v * dx(material)
        fplus += J_phi[i] * v * dx(material)
        fminus += J_phi[i] * v * dx(material)

    # Assemble bilinear and linear forms
    f.Assemble()
    fplus.Assemble()
    fminus.Assemble()
    a.Assemble()
    aplus.Assemble()
    aminus.Assemble()

    # Solve the systems for the three configurations
    start_time = time.process_time()
    rAphi_sol, rAphi_sol_plus, rAphi_sol_minus = GridFunction(Vh), GridFunction(Vhplus), GridFunction(Vhminus)
    rAphi_sol.vec.data = a.mat.Inverse(Vh.FreeDofs()) * f.vec
    rAphi_sol_plus.vec.data = aplus.mat.Inverse(Vhplus.FreeDofs()) * fplus.vec
    rAphi_sol_minus.vec.data = aminus.mat.Inverse(Vhminus.FreeDofs()) * fminus.vec

    solution_time = time.process_time()
    timings["solution"] = solution_time - start_time
    print(f"solved FEM models in {timings['solution']:.6f} seconds")

    # Reduced FE spaces for post-processing
    start_time = time.process_time()
    Vh_red = H1(mesh, definedon="Control")
    Vh_red_plus = H1(mesh_plus, definedon="Control")
    Vh_red_minus = H1(mesh_minus, definedon="Control")

    # Compute magnetic field components (Br, Bz) for each configuration
    Br_sol, Bz_sol = calculate_B_fields(Vh_red, rAphi_sol, r)
    Br_sol_plus, Bz_sol_plus = calculate_B_fields(Vh_red_plus, rAphi_sol_plus, r)
    Br_sol_minus, Bz_sol_minus = calculate_B_fields(Vh_red_minus, rAphi_sol_minus, r)



    # Evaluate field values at predefined points
    Br_vals = [Br_sol(mesh(*p)) for p in coordinates]
    Bz_vals = [Bz_sol(mesh(*p)) for p in coordinates]
    Br_vals_plus = [Br_sol_plus(mesh_plus(*p)) for p in coordinates]
    Bz_vals_plus = [Bz_sol_plus(mesh_plus(*p)) for p in coordinates]
    Br_vals_minus = [Br_sol_minus(mesh_minus(*p)) for p in coordinates]
    Bz_vals_minus = [Bz_sol_minus(mesh_minus(*p)) for p in coordinates]

    # Calculate total magnetic field magnitude
    B_vals = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals, Bz_vals)])
    B_vals_plus = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_plus, Bz_vals_plus)])
    B_vals_minus = np.array([math.sqrt(Br**2 + Bz**2) for Br, Bz in zip(Br_vals_minus, Bz_vals_minus)])

    field_computation_time = time.process_time()
    timings["field_computation"] = field_computation_time - start_time
    print(f"computed magnetic fields in {timings['field_computation']:.6f} seconds")

    # Compute metrics based on the simulation
    start_time = time.process_time()
    [f1, f2, f3] = calculate_metrics(B_vals, B_ref, B_vals_plus, B_vals_minus, n_turns, [radius] * 10, w, h, sigma_coil, I)
    metric_computation_time = time.process_time()
    timings["metric_computation"] = metric_computation_time - start_time
    print(f"computed metrics in {timings['metric_computation']:.6f} seconds")

    return Br_vals, Bz_vals, B_vals, f1, f2, f3, timings, mesh, rAphi_sol, Br_sol, Bz_sol