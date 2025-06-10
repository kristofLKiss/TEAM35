from netgen.geom2d import SplineGeometry
import numpy as np
import ngsolve
import matplotlib.pyplot as plt

from geometry.utils import append_unique_point, append_unique_line, add_rectangle_with_edge_domains

class GeometryBuilder:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mesh = None
        self.geo = None
        self.built = False

    def build(self):
        raise NotImplementedError("Define a specific geometry!")

    def plot_geometry(self):
        if not self.built:
            print("Geometry has not been built yet. Nothing to plot.")
            return

        if self.geo is None:
            print("No geometry object to plot.")
            return

        nsplines = self.geo.GetNSplines()

        plt.figure(figsize=(6,6))
        for i in range(0,nsplines):  # spline indexelés 1-től indul
            spline = self.geo.GetSpline(i)
            p1 = spline.StartPoint()
            p2 = spline.EndPoint()
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'k-')

        plt.axis("equal")
        plt.title("Geometry Lines (SplineGeometry)")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.show()

    def plot_mesh(self):
        if not self.built or self.mesh is None or self.geo is None:
            print("Geometry not built or mesh not available.")
            return

        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(6,6))

        # --- Először: geometriavonalak (splineek) ---
        for i in range(self.geo.GetNSplines()):
            spline = self.geo.GetSpline(i)
            p1 = spline.StartPoint()
            p2 = spline.EndPoint()
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],'k-',linewidth=1.1, alpha=1,label='Geometry' if i == 0 else "")

        # --- Majd: hálóélek ---
        vertex_coords = [v.point for v in self.mesh.vertices]
        for e in self.mesh.edges:
            v1,v2 = e.vertices
            x1,y1 = vertex_coords[v1.nr][0],vertex_coords[v1.nr][1]
            x2,y2 = vertex_coords[v2.nr][0],vertex_coords[v2.nr][1]
            ax.plot([x1,x2],[y1,y2],'b-',linewidth=0.4,alpha=0.6,label='Mesh Edge' if e.nr == 0 else "")

        ax.set_title("Geometry + Mesh Overlay")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        plt.show()
class LabeledSolenoidDesignSpace(GeometryBuilder):
    def __init__(self,parameters):
        super().__init__(parameters)  # ősi __init__ hívása
        # Paraméterek kinyerése és alapértelmezettek
        self.n_turns = parameters.get("n_turns",10)
        self.w = parameters.get("w",0.001)
        self.h = parameters.get("h",0.0015)
        self.insulation_thickness = parameters.get("insulation_thickness",0.00006)
        self.solh = parameters.get("solh",0.005)
        self.solw = parameters.get("solw",0.005)
        self.minradius = parameters.get("minradius",0.006)
        self.maxradius = parameters.get("maxradius",0.055)
        self.dr = parameters.get("dr",0.5e-3)
        self.maxh = parameters.get("maxh",0.001)
        self.controlledmaxh = parameters.get("controlledmaxh",0.001)
        self.wstep = parameters.get("wstep",0.0005)
        self.visu = parameters.get("visu",1)
        self.mesh = None
        self.geo = SplineGeometry()
        self.points = {}
        self.lines = set()
    def build(self):
        self.geo.SetMaterial(1, "Control")
        self.geo.SetMaterial(2, "Air")

        # Controlled region 
        pnts = [(0, 0), (0, self.solh * 1.05), (self.solw * 1.05, self.solh * 1.05), (self.solw * 1.05, 0)]
        p = [self.geo.AppendPoint(*pnt) for pnt in pnts]

        self.geo.Append(["line", p[3], p[0]], leftdomain=0, rightdomain=1, bc="bottom", maxh=self.controlledmaxh)
        self.geo.Append(["line", p[0], p[1]], leftdomain=0, rightdomain=1, bc="left", maxh=self.controlledmaxh)
        self.geo.Append(["line", p[1], p[2]], leftdomain=2, rightdomain=1, maxh=self.controlledmaxh)
        self.geo.Append(["line", p[2], p[3]], leftdomain=2, rightdomain=1, maxh=self.controlledmaxh)

        material_index = 3
        z0 = 0

        r_array = np.arange(self.minradius, self.maxradius, self.wstep)
        for i in range(self.n_turns):
            x_min = self.minradius
            for r in r_array:
                x_max = x_min + self.wstep
                y_min = z0 + self.insulation_thickness
                y_max = z0 + self.insulation_thickness + self.h

                self.geo.SetMaterial(material_index, f"T{i + 1}R{round(r, 4)}")

                if r == r_array[0]:
                    right_E = material_index + 1
                    right_W = 2
                elif r == r_array[-1]:
                    right_E = 2
                    right_W = material_index - 1
                else:
                    right_E = material_index + 1
                    right_W = material_index - 1

                add_rectangle_with_edge_domains(
                    self.geo, self.lines, self.points,
                    p1=(x_min, y_min), p2=(x_max, y_max),
                    leftdomain=material_index,
                    rightdomain_N=2, rightdomain_S=2,
                    rightdomain_W=right_W, rightdomain_E=right_E
                )

                x_min = x_max
                material_index += 1

            z0 += self.h + self.insulation_thickness

        # Extend simulation area
        max_y = max(1.6 * self.n_turns * (self.h + 2 * self.insulation_thickness), 2 * self.solh)
        sim_pnts = [(0, max_y), (0.065, max_y), (0.065, 0)]

        for pt in sim_pnts:
            p.append(append_unique_point(self.geo, *pt, self.points))

        self.geo.Append(["line", p[-1], p[3]], leftdomain=0, rightdomain=2, bc="bottom")
        self.geo.Append(["line", p[1], p[-3]], leftdomain=0, rightdomain=2, bc="left")
        self.geo.Append(["line", p[-3], p[-2]], leftdomain=0, rightdomain=2, bc="top")
        self.geo.Append(["line", p[-2], p[-1]], leftdomain=0, rightdomain=2, bc="right")
        self.mesh = ngsolve.Mesh(self.geo.GenerateMesh(maxh=self.maxh))
        self.built = True

class SolenoidWithOffsetGeometry(GeometryBuilder):
    def __init__(self, parameters):
        super().__init__(parameters)

        # Paraméterek kinyerése és alapértelmezettek
        self.n_turns = parameters.get("n_turns", 10)
        self.w = parameters.get("w", 0.001)
        self.h = parameters.get("h", 0.0015)
        self.insulation_thickness = parameters.get("insulation_thickness", 0.00006)
        self.solh = parameters.get("solh", 0.005)
        self.solw = parameters.get("solw", 0.005)
        self.radius = parameters.get("radius", None)
        self.dr = parameters.get("dr", 0.5e-3)
        self.maxh = parameters.get("maxh", 0.001)
        self.controlledmaxh = parameters.get("controlledmaxh", 0.001)
        self.visu = parameters.get("visu", 0)
        self.wstep = self.w

        if not isinstance(self.radius,(list,np.ndarray)):
            raise TypeError("'radius' must be a list or numpy array")

        if len(self.radius) != self.n_turns:
            raise ValueError(f"'radius' must have {self.n_turns} elements, got {len(self.radius)}")

        self.geo = SplineGeometry()

    def build(self):
        geo = self.geo
        n_turns = self.n_turns
        w = self.w
        h = self.h
        ins_th = self.insulation_thickness
        solh = self.solh
        solw = self.solw
        radius = self.radius
        dr = self.dr
        maxh = self.maxh
        controlledmaxh = self.controlledmaxh

        # Anyagkódok
        geo.SetMaterial(1, "Control")
        geo.SetMaterial(2, "Air")
        for i in range(n_turns):
            geo.SetMaterial(3 + i, f"T{i + 1}R{round(radius, 4)}")

        # Controlled régió (téglalap)
        p = [
            geo.AppendPoint(0, 0),
            geo.AppendPoint(0, solh * 1.05),
            geo.AppendPoint(solw * 1.05, solh * 1.05),
            geo.AppendPoint(solw * 1.05, 0),
        ]
        geo.Append(["line", p[0], p[1]], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
        geo.Append(["line", p[1], p[2]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
        geo.Append(["line", p[2], p[3]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
        geo.Append(["line", p[3], p[0]], leftdomain=0, rightdomain=1, bc="bottom", maxh=controlledmaxh)

        # Tekercsek hozzáadása
        z0 = 0
        for i in range(n_turns):
            r = radius[i]
            x_min = r + dr
            x_max = r + dr + w
            y_min = z0 + ins_th
            y_max = z0 + ins_th + h

            geo.AddRectangle(p1=(x_min, y_min), p2=(x_max, y_max),
                             leftdomain=3 + i, rightdomain=2)

            z0 += h + ins_th

        # Külső szimulációs tartomány
        max_y = max(1.6 * n_turns * (h + 2 * ins_th), 2 * solh)
        xmax = 3 * (max(radius) + w)

        p_extra = [
            geo.AppendPoint(0, max_y),
            geo.AppendPoint(xmax, max_y),
            geo.AppendPoint(xmax, 0)
        ]
        geo.Append(["line", p[1], p_extra[0]], leftdomain=0, rightdomain=2, bc="left")
        geo.Append(["line", p_extra[0], p_extra[1]], leftdomain=0, rightdomain=2, bc="top")
        geo.Append(["line", p_extra[1], p_extra[2]], leftdomain=0, rightdomain=2, bc="right")
        geo.Append(["line", p_extra[2], p[3]], leftdomain=0, rightdomain=2, bc="bottom")

        # Meshing
        print("Meshing")
        self.mesh = ngsolve.Mesh(geo.GenerateMesh(maxh=maxh))
        print("Meshing done")
        self.built = True


class UniformRadiusSolenoidGeometry(GeometryBuilder):
    def __init__(self, parameters):
        super().__init__(parameters)

        self.n_turns = parameters.get("n_turns", 10)
        self.w = parameters.get("w", 0.01)
        self.h = parameters.get("h", 0.015)
        self.insulation_thickness = parameters.get("insulation_thickness", 0.00006)
        self.solh = parameters.get("solh", 0.005)
        self.solw = parameters.get("solw", 0.005)
        self.radius = parameters.get("radius", 0.01)
        self.dr = parameters.get("dr", 0.0)
        self.maxh = parameters.get("maxh", 0.001)
        self.controlledmaxh = parameters.get("controlledmaxh", 0.001)
        self.visu = parameters.get("visu", 0)
        self.wstep = self.w
        self.geo = SplineGeometry()
        self.mesh = None

    def build(self):
        geo = self.geo
        n_turns = self.n_turns
        w = self.w
        h = self.h
        ins_th = self.insulation_thickness
        solh = self.solh
        solw = self.solw
        r = self.radius
        dr = self.dr
        maxh = self.maxh
        controlledmaxh = self.controlledmaxh

        # Anyagok
        geo.SetMaterial(1, "Control")
        geo.SetMaterial(2, "Air")
        for i in range(n_turns):
            geo.SetMaterial(3 + i, f"T{i + 1}R{round(r, 4)}")

        # Kontrolltéglalap
        p = [
            geo.AppendPoint(0, 0),
            geo.AppendPoint(0, solh * 1.05),
            geo.AppendPoint(solw * 1.05, solh * 1.05),
            geo.AppendPoint(solw * 1.05, 0)
        ]
        geo.Append(["line", p[0], p[1]], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
        geo.Append(["line", p[1], p[2]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
        geo.Append(["line", p[2], p[3]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
        geo.Append(["line", p[3], p[0]], leftdomain=0, rightdomain=1, bc="bottom", maxh=controlledmaxh)

        # Tekercsek
        z0 = 0
        x_min = r + dr
        x_max = x_min + w
        for i in range(n_turns):
            y_min = z0 + ins_th
            y_max = z0 + ins_th + h

            geo.AddRectangle(p1=(x_min, y_min), p2=(x_max, y_max),
                             leftdomain=3 + i, rightdomain=2)

            z0 += h + ins_th

        # Külső szimulációs tartomány
        max_y = max(1.6 * n_turns * (h + 2 * ins_th), 2 * solh)
        xmax = 3 * (r + dr + w)

        p_extra = [
            geo.AppendPoint(0, max_y),
            geo.AppendPoint(xmax, max_y),
            geo.AppendPoint(xmax, 0)
        ]

        geo.Append(["line", p[1], p_extra[0]], leftdomain=0, rightdomain=2, bc="left")
        geo.Append(["line", p_extra[0], p_extra[1]], leftdomain=0, rightdomain=2, bc="top")
        geo.Append(["line", p_extra[1], p_extra[2]], leftdomain=0, rightdomain=2, bc="right")
        geo.Append(["line", p_extra[2], p[3]], leftdomain=0, rightdomain=2, bc="bottom")

        self.mesh = ngsolve.Mesh(geo.GenerateMesh(maxh=maxh))
        self.built = True
