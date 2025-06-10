from abc import ABC, abstractmethod
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt

class NGSolveProblem(ABC):
    def __init__(self, geometry):
        if not geometry.built:
            raise ValueError("Geometry must be built before PDE problem is constructed.")
        self.mesh = geometry.mesh
        self.geo = geometry.geo
        self.n_turns = geometry.n_turns
        self.solution = None
        self.source_terms = None
        self.source_materials = None

    @abstractmethod
    def setup_bilinearform(self):
        pass
    def setup_linearform(self):
        pass
    @abstractmethod
    def setup_materials(self):
        pass
    def setup_sources(self):
        pass



    @abstractmethod
    def solve(self):
        pass


class MagnetostaticProblem(NGSolveProblem):
    def __init__(self, geometry, mode="2D-planar"):
        super().__init__(geometry)
        self.mode = mode
        self.geoparam = geometry
        self.V = None
        self.a = None
        self.f = None
        self.u = None
        self.v = None
        self.r = None # only in 2d-axi


    def setup_bilinearform(self,dirichlet_bcs: str,assemble=True):
        """
        Setup the left-hand side of the variational formulation for the problem.

        Parameters
        ----------
        dirichlet_bcs : str
            Pipe-separated string of boundary names to be used as Dirichlet conditions (e.g. "left|right|top").
        assemble : bool
            If True, assemble the bilinear form after setup.
        """
        assert isinstance(dirichlet_bcs,str) and dirichlet_bcs.strip() != "", \
            "Dirichlet boundary condition string must be a non-empty string."

        dirichlet_list = dirichlet_bcs.strip().split("|")
        mesh_boundaries = set(self.mesh.GetBoundaries())
        for bc in dirichlet_list:
            assert bc in mesh_boundaries,f"Boundary '{bc}' not found in mesh boundaries: {mesh_boundaries}"

        self.V = H1(self.mesh,order=2,dirichlet=dirichlet_bcs)
        self.u = self.V.TrialFunction()
        self.v = self.V.TestFunction()

        if self.mode == "2D-planar":
            self.a = BilinearForm(self.V)
            self.a += self.nu * grad(self.u) * grad(self.v) * dx

        elif self.mode == "2D-axi":
            self.r = x + 1e-8  # avoid division by zero
            self.a = BilinearForm(self.V)
            self.a += self.nu * grad(self.u) * grad(self.v) * (1 / self.r) * dx

        elif self.mode == "3D":
            self.V = None
            self.a = None
            self.f = None
            print("3D mode is not implemented yet.")
            return

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if assemble:
            self.a.Assemble()


    def setup_linearform(self,assemble=True):
        """
        Setup the rhs of the variational formulation for the problem.

        Parameters
        ----------
        assemble : bool
           If True, assemble the bilinear and linear forms after setup.
        """
        if self.mode == "2D-planar":
            self.f = LinearForm(self.V)
            for material,source in self.source_terms:
                self.f += source * self.v * dx(material)

        elif self.mode == "2D-axi":
            self.f = LinearForm(self.V)
            for material,source in self.source_terms:
                self.f += source * self.v * dx(material)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if assemble and self.V:
            self.f.Assemble()
    def setup_materials(self,mu_list: dict):
        """
        Setup linear reluctivity and source terms for the PDE. TODO: nonlinearity

        Parameters
        ----------
        mu_list : dict
            Dictionary mapping material names to their relative permeability, e.g. {"air": 1.0, "coil": 1000.0}.
       """

        # Check that all materials in the mesh are defined in mu_list
        material_names = set(self.mesh.GetMaterials())
        mu_keys = set(mu_list.keys())
        missing = material_names - mu_keys
        if missing:
            raise ValueError(f"Missing permeability definitions for materials: {missing}")

        # Create CoefficientFunction for reluctivity (nu)
        self.nu = CoefficientFunction([1/mu_list[mat] for mat in self.mesh.GetMaterials()])

    def setup_sources(self,source_materials: list, sources):
        """
        Setup sources of the model by listing each material to be used as a source and their respective current density.

        Parameters
        ----------
        source_materials : list of str
            List of material names where source terms (current density) are applied.
        sources : Parameter or list of Parameter
            A single source term (applied to all materials), or a list of Parameters matching source_materials.
        """
        if isinstance(sources,list):
            if len(sources) != len(source_materials):
                raise ValueError("Length of 'sources' must match length of 'source_materials'.")
            self.source_terms = list(zip(source_materials,sources))
        else:
            # Apply the same source to all listed materials
            self.source_terms = [(mat,sources) for mat in source_materials]


    def solve(self):
        if not self.V:
            print("No solution space defined — skipping solve.")
            return

        self.solution = GridFunction(self.V)
        self.solution.Set(0)
        self.solution.vec.data = self.a.mat.Inverse(self.V.FreeDofs()) * self.f.vec

    def compute_B_fields(self,points):
        """
        Postprocessing: Compute Bx and By.
        """
        if self.mode == "2D-axi":
            Aphi = GridFunction(self.V)
            Aphi.Set(self.solution / self.r)
            Bx = GridFunction(self.V)
            By = GridFunction(self.V)
            Bx.Set(-grad(Aphi)[1])
            By.Set((1 / self.r) * grad(self.solution)[0])

        if self.mode == "2D-planar":
            Bx = GridFunction(self.V)
            By = GridFunction(self.V)
            Bx.Set(-grad(self.solution)[1])
            By.Set(grad(self.solution)[0])

        self.Bx = [Bx(self.mesh(*p)) for p in points]
        self.By = [By(self.mesh(*p)) for p in points]

