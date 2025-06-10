import numpy as np

# Preprocessing functions
def compute_Jphi(I, w, h):
    '''
    Áramsűrűség számítása áramból és geometriából
    '''
    return I / (w * h)

# Postprocessing functions

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
