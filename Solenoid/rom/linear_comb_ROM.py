import numpy as np
from ngsolve import *
from problem.NGSolveProblem import MagnetostaticProblem
from opti.utils import *
from rom.utils import *

class linear_comb_ROM:
    def __init__(self, problem: MagnetostaticProblem):
        if problem.mode != "2D-axi":
            raise NotImplementedError("Only 2D-axi ROM is currently implemented.")
        self.problem = problem

    def create_sourcebasis(self,source_materials,J,reducedSpaceOutputDomain):
        """
        Create basis solutions for static magnetic vector potential for multiple sources
        Parameters
        ----------
        source_materials : str list
            for each material on the list a single unit source will be applied
        J : float
            J is the source basis [A/m^2],
                1/A_cond when initializing with a unit current, with A_cond being the conductor cross-section
                u/(R*A_cond) when initializing with unit voltage u and R being the conductor resistance
        reducedSpaceOutputDomain : FESpace defined on a subdomain of problem.mesh
            If left empty the result will be the whole domain of problem.mesh
        """
        rA = []
        if reducedSpaceOutputDomain is not None:
            self.solution_gridfunction = GridFunction(reducedSpaceOutputDomain)
            self.OutputDomain = reducedSpaceOutputDomain
            self.onreducedDomain = 1
            for i,material in enumerate(source_materials):
                # print(f"Index: {i}, Material: {material}")
                # Define linear form
                self.problem.setup_sources([material],J)
                self.problem.setup_linearform(assemble=True)
                # print(max(self.problem.f.vec))
                # Solve system
                self.problem.solve()
                # Reduced solution on a reduced subspace
                self.solution_gridfunction.Set(self.problem.solution)
                # print(max(self.problem.solution.vec.data))
                # Store results
                # print(self.solution_gridfunction.vec.data)
                rA.append(self.solution_gridfunction.vec.data)
                # print(np.array(rA))
            #

        else:
            self.onreducedDomain = 0
            self.solution_gridfunction = GridFunction(self.problem.V)
            self.OutputDomain = self.problem.V
            for i,material in enumerate(source_materials):
                # print(f"Index: {i}, Material: {material}")
                # Define linear form
                self.problem.setup_sources([material],J)
                self.problem.setup_linearform(assemble=True)
                # Solve system
                self.problem.solve()
                # Store results
                rA.append(self.problem.solution)

        self.material_lookup = preprocess_coil_materials(source_materials)
        # Determine the size of the full reduced space vector based on the lookup table
        self.max_current_index = max(self.material_lookup.values()) + 1
        self.Amat = np.array(rA).copy()
        self.solution_gridfunction.Set(0)


    def calc_a_with_ROM(self,ri_vec,I):
        """
           Compute the reduced-order magnetic vector potential solution using the ROM basis.

           Parameters
           ----------
           ri_vec : list or array of float
               Radial positions [m] of the coil turns (inner position of each turn).
           I : list or array of float
               Current amplitudes [A] for each coil turn.

           Notes
           -----
           This method computes a linear combination of the precomputed ROM basis solutions
           (stored in self.Amat) based on the geometry-dependent material lookup and input
           current distribution. The result is stored in self.solution_gridfunction.
        """
        i_vec = list(range(1, self.problem.n_turns + 1))
        current_vector = np.zeros(self.max_current_index)
        # Iterálás az összes menet és sugár felett
        for turn_idx,(i,ri) in enumerate(zip(i_vec,ri_vec)):
            # Find first index
            start_index = self.material_lookup.get((i,round(ri,4)))
            if start_index is None:
                raise ValueError(f"Material for T{i}R{ri:.4f} not found.")
            # Fill slices with current
            num_steps = int(self.problem.geoparam.w / self.problem.geoparam.wstep)  # Required no of slices
            for step in range(num_steps):
                current_radius = round(ri + step * self.problem.geoparam.wstep,10)
                idx = self.material_lookup.get((i,current_radius))
                if idx is not None:
                    current_vector[idx] = I[turn_idx]  # Assign turn current
            self.solution_gridfunction.vec.data = np.dot(current_vector,self.Amat)
    def compute_B_fields_with_ROM(self,points):
        """
        Postprocessing: Compute Bx and By for reduced solution.
        """
        if self.problem.mode == "2D-axi":
            Aphi = GridFunction(self.OutputDomain)
            Aphi.Set(self.solution_gridfunction / self.problem.r)
            Bx = GridFunction(self.OutputDomain)
            By = GridFunction(self.OutputDomain)
            Bx.Set(-grad(Aphi)[1])
            By.Set((1 / self.problem.r) * grad(self.solution_gridfunction)[0])

        if self.problem.mode == "2D-planar":
            Bx = GridFunction(self.OutputDomain)
            By = GridFunction(self.OutputDomain)
            Bx.Set(-grad(self.solution_gridfunction)[1])
            By.Set(grad(self.solution_gridfunction)[0])

        Bx = [Bx(self.problem.mesh(*p)) for p in points]
        By = [By(self.problem.mesh(*p)) for p in points]
        return Bx,By