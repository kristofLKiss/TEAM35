from netgen.meshing import *
import ngsolve
# import netgen.gui
from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import matplotlib.pyplot as plt

def create_ngsolve_mesh(xmin=0,xmax=3,Nx=10,ymin=-1,ymax=1,Ny=10,quads=True):
    """
    Function to create an NGSolve mesh object with the specified parameters.
    Parameters:
    xmin, xmax: float : X-axis range of the mesh.
    Nx: int : Number of divisions along the X-axis.
    ymin, ymax: float : Y-axis range of the mesh.
    Ny: int : Number of divisions along the Y-axis.
    quads: bool : Whether to create quadrilateral elements (True) or triangular elements (False).
    Returns:
    ngsolve.Mesh : The created NGSolve mesh object.
    """

    ngmesh = Mesh()
    ngmesh.dim = 2

    # Generate mesh points
    pnums = []
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            pnums.append(ngmesh.Add(MeshPoint(Pnt(xmin + (xmax - xmin) * i / Nx,
                                                  ymin + (ymax - ymin) * j / Ny,0))))

    # Add face descriptor and material for first region
    ngmesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
    ngmesh.SetMaterial(1,"mat")

    # Add elements (quadrilaterals or triangles)
    for j in range(Nx):
        for i in range(Ny):
            if quads:
                ngmesh.Add(Element2D(1,[pnums[i + j * (Ny + 1)],
                                        pnums[i + (j + 1) * (Ny + 1)],
                                        pnums[i + 1 + (j + 1) * (Ny + 1)],
                                        pnums[i + 1 + j * (Ny + 1)]]))
            else:
                # Add triangular elements
                ngmesh.Add(Element2D(1,[pnums[i + j * (Ny + 1)],
                                        pnums[i + (j + 1) * (Ny + 1)],
                                        pnums[i + 1 + j * (Ny + 1)]]))
                ngmesh.Add(Element2D(1,[pnums[i + (j + 1) * (Ny + 1)],
                                        pnums[i + 1 + (j + 1) * (Ny + 1)],
                                        pnums[i + 1 + j * (Ny + 1)]]))

    cnt_ident = 1
    # Horizontal boundaries and point identifications
    for i in range(Nx):
        ngmesh.Add(Element1D([pnums[Ny + i * (Ny + 1)],pnums[Ny + (i + 1) * (Ny + 1)]],index=3)) #top
        ngmesh.Add(Element1D([pnums[0 + i * (Ny + 1)],pnums[0 + (i + 1) * (Ny + 1)]],index=1))
        # ngmesh.AddPointIdentification(pnums[0 + i * (Ny + 1)],pnums[Ny + i * (Ny + 1)],
        #                               identnr=cnt_ident,type=2)

    # ngmesh.AddPointIdentification(pnums[0 + Nx * (Ny + 1)],pnums[Ny + Nx * (Ny + 1)],
    #                               identnr=cnt_ident,type=2)
    cnt_ident += 1
    # Vertical boundaries and point identifications
    for i in range(Ny):
        ngmesh.Add(Element1D([pnums[i],pnums[i + 1]],index=4))
        ngmesh.Add(Element1D([pnums[i + Nx * (Ny + 1)],pnums[i + 1 + Nx * (Ny + 1)]],index=2))
        # ngmesh.AddPointIdentification(pnums[i],pnums[i + Nx * (Ny + 1)],
        #                               identnr=cnt_ident,type=2)

    # ngmesh.AddPointIdentification(pnums[Ny],pnums[Ny + Nx * (Ny + 1)],
    #                               identnr=cnt_ident,type=2)


    # Add  boundary names
    ngmesh.SetBCName(0,"left")
    ngmesh.SetBCName(1,"top")
    ngmesh.SetBCName(2,"right")
    ngmesh.SetBCName(3,"bottom")
    # ngmesh.SetBCName(4,"left")
    # Save the mesh in .vol format (optional, can be removed if not needed)
    # ngmesh.Save('quad_per.vol')

    # Convert Netgen mesh to NGSolve mesh
    mesh = ngsolve.Mesh(ngmesh)

    return mesh


def generate_ngsolve_geometry_and_mesh(n_turns=10, w=0.01, h=0.015, solh=0.05, solw=0.05, radius=0.1, maxh=0.01, controlledmaxh=0.01,visu=1):
    geo = SplineGeometry()
    pnts = [None] * 4
    lines = [None] * 4
     # Domains
    geo.SetMaterial (1, "Control")
    geo.SetMaterial (2, "Air")
    geo.SetMaterial (3, "T1")
    geo.SetMaterial (4, "T2")
    geo.SetMaterial (5, "T3")
    geo.SetMaterial (6, "T4")
    geo.SetMaterial (7, "T5")
    geo.SetMaterial (8, "T6")
    geo.SetMaterial (9, "T7")
    geo.SetMaterial (10, "T8")
    geo.SetMaterial (11, "T9")
    geo.SetMaterial (12, "T10")
    # Controlled region
    pnts[0], pnts[1], pnts[2], pnts[3] = [(x, y) for x, y in [(0, -solh), (0, solh), (solw, solh), (solw, -solh)]]
    p = [ geo.AppendPoint(*pnt) for pnt in pnts ]
    lines[0] = ["line", p[0], p[1]]
    lines[1] = ["line", p[1], p[2]]
    lines[2] = ["line", p[2], p[3]]
    lines[3] = ["line", p[3], p[0]]
    geo.Append (lines[0], leftdomain=0, rightdomain=1, bc="left",maxh=controlledmaxh)
    geo.Append (lines[1], leftdomain=2, rightdomain=1,maxh=controlledmaxh)
    geo.Append (lines[2], leftdomain=2, rightdomain=1,maxh=controlledmaxh)
    geo.Append (lines[3], leftdomain=2, rightdomain=1,maxh=controlledmaxh)


    # Solenoid
    z0 = float(0);  # -(h + gap) * n_turns / 2  # Starting position for the turns
    for i in range(n_turns):
        # print(z0)
        x_min = radius
        x_max = radius + w
        y_min = z0
        y_max = z0 + h

        if i>1: # azonosan egyenlő pontok, ha i==0, még nem él a szabály, ha i==1
            # Felfelé
            pnts.append((x_min,y_max))  # 10+4*(i-1)
            pnts.append((x_max,y_max))  # 11+4*(i-1)
            # Lefelé (így változik a körüljárási irány!! Leftdomain, Rightdomain csere!)
            pnts.append((x_min,-y_max))  # 12+4*(i-1)
            pnts.append((x_max,-y_max))  # 13 +4*(i-1)
            # Iteráció a pontokon, és mindegyik külön hozzáadása
            for k in range(10,14):  # k = 10-től 13-ig
                point = pnts[k + (4 * (i-1))]
                p.append(geo.AppendPoint(*point))
            # Vonalak hozzáadása
            lines.append(["line",p[10+(i-2)*4],p[14+(i-2)*4]])  # 17+
            lines.append(["line",p[14 + (i - 2) * 4],p[15 + (i - 2) * 4]])  # 18+
            lines.append(["line",p[15 + (i - 2) * 4],p[11 + (i - 2) * 4]])  # 19+
            # lenti vonalak
            lines.append(["line",p[12 + (i - 2) * 4],p[16 + (i - 2) * 4]])  # 20+
            lines.append(["line",p[16 + (i - 2) * 4],p[17 + (i - 2) * 4]])  # 21+
            lines.append(["line",p[17 + (i - 2) * 4],p[13 + (i - 2) * 4]])  # 22+

            geo.Append(lines[17+(i-2)*6],leftdomain=2,rightdomain=5+(i-2))

            if i < n_turns-1:
                geo.Append(lines[18 + (i - 2) * 6],leftdomain=6+(i-2),rightdomain=5 + (i - 2))
            else:  # a legfelső vonal felett levegő van
                geo.Append(lines[18 + (i - 2) * 6],leftdomain=2,rightdomain=5 + (i - 2))
                print("could close")
            geo.Append(lines[19 + (i - 2) * 6],leftdomain=2,rightdomain=5 + (i - 2))
            # lenti vonalak
            geo.Append(lines[20 + (i - 2) * 6],leftdomain=5 + (i - 2),rightdomain=2)
            if i < n_turns-1:
                geo.Append(lines[21 + (i - 2) * 6],leftdomain=5 + (i - 2),rightdomain=6 + (i - 2))
            else:  # a legalsó vonal alatt levegő van
                geo.Append(lines[21 + (i - 2) * 6],leftdomain=5 + (i - 2),rightdomain=2)
            geo.Append(lines[22 + (i - 2) * 6],leftdomain=5 + (i - 2),rightdomain=2)
            print(5+(i-2))
        elif i==1:
            # Felfelé
            pnts.append((x_min,y_max))  # 10+4*(i-1)
            pnts.append((x_max,y_max))  # 11+4*(i-1)
            # Lefelé (így változik a körüljárási irány!! Leftdomain, Rightdomain csere!)
            pnts.append((x_min,-y_max))  # 12+4*(i-1)
            pnts.append((x_max,-y_max))  # 13 +4*(i-1)
            for k in range(10,14):  # k = 10-től 13-ig
                point = pnts[k]
                p.append(geo.AppendPoint(*point))
            lines.append(["line",p[5],p[10]])  # 11
            lines.append(["line",p[10],p[11]])  # 12
            lines.append(["line",p[11],p[6]])  # 13
            lines.append(["line",p[8],p[12]])  # 14
            lines.append(["line",p[12],p[13]])  # 15
            lines.append(["line",p[13],p[9]])  # 16
            geo.Append(lines[11],leftdomain=2,rightdomain=4)
            geo.Append(lines[12],leftdomain=5,rightdomain=4)
            geo.Append(lines[13],leftdomain=2,rightdomain=4)
            # Fordul a körüljárási irány, left <-> right
            geo.Append(lines[14],leftdomain=4,rightdomain=2)
            geo.Append(lines[15],leftdomain=4,rightdomain=5)
            geo.Append(lines[16],leftdomain=4,rightdomain=2)

        elif i==0:
            pnts.append((x_min,y_min))  # 4
            pnts.append((x_min,y_max))  # 5
            pnts.append((x_max,y_max))  # 6
            pnts.append((x_max,y_min))  # 7
            # Lefelé (így változik a körüljárási irány!! Leftdomain, Rightdomain csere!)
            # A közös vonal azonosan egyenlő
            pnts.append((x_min,-y_max))  # 8
            pnts.append((x_max,-y_max))  # 9
            # Iteráció a pontokon, és mindegyik külön hozzáadása
            for k in range(4,10):  # k = 4-től 9-ig
                point = pnts[k]
                p.append(geo.AppendPoint(*point))
            lines.append(["line", p[4], p[5]]) # [4]
            lines.append(["line",p[5],p[6]]) #5
            lines.append(["line",p[6],p[7]])#6
            lines.append(["line",p[7],p[4]]) #7 közös vonal az alsó és a felső
            lines.append(["line",p[4],p[8]]) #8
            lines.append(["line",p[8],p[9]]) #9
            lines.append(["line",p[9],p[7]]) #10
            geo.Append(lines[4],leftdomain=2,rightdomain=3)
            geo.Append(lines[5],leftdomain=4,rightdomain=3)
            geo.Append(lines[6],leftdomain=2,rightdomain=3)
            geo.Append(lines[7],leftdomain=3,rightdomain=3)
            # Fordul a körüljárási irány, left <-> right
            geo.Append(lines[8],leftdomain=3,rightdomain=2)
            geo.Append(lines[9],leftdomain=3,rightdomain=4)
            geo.Append(lines[10],leftdomain=3,rightdomain=2)
        z0 += h

    # Simulation area
    contidxLines = len(lines)
    contidxPts = len(pnts)
    pnts.append((0,max((2*n_turns*h),2*solh)))
    pnts.append((2*(radius+w),max((2*n_turns*h),2*solh)))
    pnts.append((2*(radius+w),-max((2*n_turns*h),2*solh)))
    pnts.append((0,-max((2*n_turns*h),2*solh)))
    p.append(geo.AppendPoint(*pnts[contidxPts]))
    p.append(geo.AppendPoint(*pnts[contidxPts+1]))
    p.append(geo.AppendPoint(*pnts[contidxPts+2]))
    p.append(geo.AppendPoint(*pnts[contidxPts+3]))

    lines.append(["line",p[1],p[contidxPts]])
    geo.Append(lines[contidxLines],leftdomain=0,rightdomain=2,bc="left")
    lines.append(["line",p[contidxPts],p[contidxPts+1]])
    geo.Append(lines[contidxLines+1],leftdomain=0,rightdomain=2,bc="top")
    lines.append(["line",p[contidxPts+1],p[contidxPts+2]])
    geo.Append(lines[contidxLines+2],leftdomain=0,rightdomain=2,bc="right")
    lines.append(["line",p[contidxPts+2],p[contidxPts+3]])
    geo.Append(lines[contidxLines+3],leftdomain=0,rightdomain=2,bc="bottom")
    lines.append(["line",p[contidxPts+3],p[0]])
    geo.Append(lines[contidxLines+4],leftdomain=0,rightdomain=2,bc="left")

    # Plot inicializálása
    plt.figure(figsize=(8, 8))
    if visu == 1:
        # Pontok ábrázolása (kék színnel)
        x_coords, y_coords = zip(*pnts)
        plt.scatter(x_coords, y_coords, color='blue', label='Points')
        for idx, (x, y) in enumerate(pnts):
            plt.text(x, y, f'{idx}', fontsize=10, color='black', ha='right')
        # Vonalszakaszok ábrázolása (piros színnel)
        for line in lines:
            _, start_idx, end_idx = line
            x = [pnts[start_idx][0], pnts[end_idx][0]]
            y = [pnts[start_idx][1], pnts[end_idx][1]]
            plt.plot(x, y, color='red', label='Lines' if line == lines[0] else "")

        # Címek és beállítások
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Pontok és vonalak ábrázolása')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.axis('equal')  # Megfelelő arányok biztosítása
        plt.show()
    ngmesh = geo.GenerateMesh(maxh=maxh) # netgen háló
    mesh = ngsolve.Mesh(ngmesh) # ngsolve háló
    return mesh

def generate_ngsolve_geometry_and_mesh_w_ins(n_turns=10, w=0.01, h=0.015, insulation_thickness=0.00006, solh=0.05, solw=0.05, radius=0.1, maxh=0.001, controlledmaxh=0.001,visu=1):
    geo = SplineGeometry()
    pnts = [None] * 4
    lines = [None] * 4
     # Domains
    geo.SetMaterial (1, "Control")
    geo.SetMaterial (2, "Air")
    geo.SetMaterial (3, "T1")
    geo.SetMaterial (4, "T2")
    geo.SetMaterial (5, "T3")
    geo.SetMaterial (6, "T4")
    geo.SetMaterial (7, "T5")
    geo.SetMaterial (8, "T6")
    geo.SetMaterial (9, "T7")
    geo.SetMaterial (10, "T8")
    geo.SetMaterial (11, "T9")
    geo.SetMaterial (12, "T10")
    # Controlled region
    pnts[0], pnts[1], pnts[2], pnts[3] = [(x, y) for x, y in [(0, -solh), (0, solh), (solw, solh), (solw, -solh)]]
    p = [ geo.AppendPoint(*pnt) for pnt in pnts ]
    lines[0] = ["line", p[0], p[1]]
    lines[1] = ["line", p[1], p[2]]
    lines[2] = ["line", p[2], p[3]]
    lines[3] = ["line", p[3], p[0]]
    geo.Append (lines[0], leftdomain=0, rightdomain=1, bc="left",maxh=controlledmaxh)
    geo.Append (lines[1], leftdomain=2, rightdomain=1,maxh=controlledmaxh)
    geo.Append (lines[2], leftdomain=2, rightdomain=1,maxh=controlledmaxh)
    geo.Append (lines[3], leftdomain=2, rightdomain=1,maxh=controlledmaxh)


    # Solenoid
    z0 = float(0);  # -(h + gap) * n_turns / 2  # Starting position for the turns
    for i in range(n_turns):
        # print(z0)
        x_min = radius
        x_max = radius + w
        y_min = z0 + insulation_thickness
        y_max = z0 + insulation_thickness + h
        geo.AddRectangle(p1=(x_min,y_min),
                         p2=(x_max,y_max),
                         leftdomain=3+i,
                         rightdomain=2)
        geo.AddRectangle(p1=(x_min,-y_max),
                         p2=(x_max,-y_min),
                         leftdomain=3+i,
                         rightdomain=2)
        z0 += (h + insulation_thickness)
    # Simulation area
    contidxLines = len(lines)
    contidxPts = len(pnts)
    pnts.append((0,max((2*n_turns*(h+2*insulation_thickness)),2*solh)))
    pnts.append((2*(radius+w),max((2*n_turns*(h+2*insulation_thickness)),2*solh)))
    pnts.append((2*(radius+w),-max((2*n_turns*(h+2*insulation_thickness)),2*solh)))
    pnts.append((0,-max((2*n_turns*(h+2*insulation_thickness)),2*solh)))
    p.append(geo.AppendPoint(*pnts[contidxPts]))
    p.append(geo.AppendPoint(*pnts[contidxPts+1]))
    p.append(geo.AppendPoint(*pnts[contidxPts+2]))
    p.append(geo.AppendPoint(*pnts[contidxPts+3]))

    lines.append(["line",p[1],p[contidxPts]])
    geo.Append(lines[contidxLines],leftdomain=0,rightdomain=2,bc="left")
    lines.append(["line",p[contidxPts],p[contidxPts+1]])
    geo.Append(lines[contidxLines+1],leftdomain=0,rightdomain=2,bc="top")
    lines.append(["line",p[contidxPts+1],p[contidxPts+2]])
    geo.Append(lines[contidxLines+2],leftdomain=0,rightdomain=2,bc="right")
    lines.append(["line",p[contidxPts+2],p[contidxPts+3]])
    geo.Append(lines[contidxLines+3],leftdomain=0,rightdomain=2,bc="bottom")
    lines.append(["line",p[contidxPts+3],p[0]])
    geo.Append(lines[contidxLines+4],leftdomain=0,rightdomain=2,bc="left")
    ngmesh = geo.GenerateMesh(maxh=maxh) # netgen háló
    mesh = ngsolve.Mesh(ngmesh) # ngsolve háló
    return mesh,geo

from netgen.occ import *
import time

# NEM HASZNÁLT:
def generate_3d_ngsolve_geometry_and_mesh_w_ins(n_turns=10, w=0.01, h=0.015, insulation_thickness=0.00006, solh=0.05, solw=0.05, radius=0.1, maxh=0.1, controlledmaxh=0.001,visu=1):
    start_time = time.time()
    print("building geometry")
    z0 = 0
    # geo = OCCGeometry()
    coilpos = []
    coilneg = []
    controlled_domain = Cylinder(Pnt(0,0,-solh),Z,r=solw,h=2*solh)
    for idx in range(n_turns):
        x_min = radius
        x_max = radius + w
        y_min = z0 + insulation_thickness
        y_max = z0 + insulation_thickness + h
        cyl = Cylinder(Pnt(0,0,y_min),Z,r=x_max,h=h)
        cyl2 = Cylinder(Pnt(0,0,y_min),Z,r=x_min,h=h)
        coilpos.append(cyl-cyl2)
        cyl3 = Cylinder(Pnt(0,0,-y_max),Z,r=x_max,h=h)
        cyl4 = Cylinder(Pnt(0,0,-y_max),Z,r=x_min,h=h)
        coilneg.append(cyl3-cyl4)
        # print(f"T{idx+1}")
        coilpos[idx].mat(f"T{idx+1}")
        coilneg[idx].mat(f"T{idx+1}")
        z0 += (h + insulation_thickness)
    box = Box((-2 * (radius + w),
              -2 * (radius + w),
             -max((2 * n_turns * (h + 2 * insulation_thickness),2 * solh))),
              (2 * (radius + w),
              2 * (radius + w),
              max((2 * n_turns * (h + 2 * insulation_thickness),2 * solh))))
    box.faces.name = "outer"

    coil_combined = Glue([*coilpos,*coilneg])
    air = box-coil_combined-controlled_domain
    air.mat("Air")

    final_geometry = Glue([air,coil_combined,controlled_domain])
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"created geometry in {elapsed_time:.6f} seconds")
    print("gluing occ geometry...")
    start_time = time.time()
    # geo = OCCGeometry(final_geometry)
    end_time=time.time()
    elapsed_time = end_time - start_time
    print(f"glued occ geometry in {elapsed_time:.6f} seconds")

    print("creating ngmesh...")
    start_time = time.time()
    ngmesh = OCCGeometry(final_geometry).GenerateMesh(maxh=3)
    end_time=time.time()
    elapsed_time = end_time - start_time
    print(f"glued occ geometry in {elapsed_time:.6f} seconds")

    print("saving geometry...")
    ngmesh.Save("fullcoil.vol")
    mesh = ngsolve.Mesh(ngmesh)
    return mesh

# NEM HASZNÁLT:
def generate_ngsolve_geometry_and_mesh_w_ins_offset(n_turns=10, w=0.01, h=0.015, insulation_thickness=0.00006,
                                                    solh=0.05, solw=0.05, radius=0.1, dr=0.5e-3,
                                                    maxh=0.001, controlledmaxh=0.001, visu=1):
    geo = SplineGeometry()
    pnts = [None] * 4
    lines = [None] * 4

    # Domains
    geo.SetMaterial(1, "Control")
    geo.SetMaterial(2, "Air")
    for i in range(10):
        geo.SetMaterial(3 + i, f"T{i + 1}")

    # Controlled region
    pnts[0], pnts[1], pnts[2], pnts[3] = [(x, y) for x, y in [(0, -solh), (0, solh), (solw, solh), (solw, -solh)]]
    p = [geo.AppendPoint(*pnt) for pnt in pnts]
    lines[0] = ["line", p[0], p[1]]
    lines[1] = ["line", p[1], p[2]]
    lines[2] = ["line", p[2], p[3]]
    lines[3] = ["line", p[3], p[0]]
    geo.Append(lines[0], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
    geo.Append(lines[1], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[2], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[3], leftdomain=2, rightdomain=1, maxh=controlledmaxh)

    # Solenoid
    z0 = float(0)  # Starting position for the turns
    for i in range(n_turns):
        x_min = radius + dr
        x_max = radius + dr + w
        y_min = z0 + insulation_thickness
        y_max = z0 + insulation_thickness + h

        geo.AddRectangle(p1=(x_min, y_min),
                         p2=(x_max, y_max),
                         leftdomain=3 + i,
                         rightdomain=2)
        geo.AddRectangle(p1=(x_min, -y_max),
                         p2=(x_max, -y_min),
                         leftdomain=3 + i,
                         rightdomain=2)
        z0 += (h + insulation_thickness)

    # Simulation area
    pnts.append((0, max((2 * n_turns * (h + 2 * insulation_thickness)), 2 * solh)))
    pnts.append((2 * (radius + dr + w), max((2 * n_turns * (h + 2 * insulation_thickness)), 2 * solh)))
    pnts.append((2 * (radius + dr + w), -max((2 * n_turns * (h + 2 * insulation_thickness)), 2 * solh)))
    pnts.append((0, -max((2 * n_turns * (h + 2 * insulation_thickness)), 2 * solh)))

    p.append(geo.AppendPoint(*pnts[-4]))
    p.append(geo.AppendPoint(*pnts[-3]))
    p.append(geo.AppendPoint(*pnts[-2]))
    p.append(geo.AppendPoint(*pnts[-1]))

    geo.Append(["line", p[1], p[-4]], leftdomain=0, rightdomain=2, bc="left")
    geo.Append(["line", p[-4], p[-3]], leftdomain=0, rightdomain=2, bc="top")
    geo.Append(["line", p[-3], p[-2]], leftdomain=0, rightdomain=2, bc="right")
    geo.Append(["line", p[-2], p[-1]], leftdomain=0, rightdomain=2, bc="bottom")
    geo.Append(["line", p[-1], p[0]], leftdomain=0, rightdomain=2, bc="left")



    # Generate meshes
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ngsolve.Mesh(ngmesh)

    return mesh, geo

def generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane(n_turns=10, w=0.01, h=0.015, insulation_thickness=0.00006,
                                                    solh=0.005, solw=0.005, radius=0.01, dr=0,
                                                    maxh=0.001, controlledmaxh=0.001, visu=0):
    geo = SplineGeometry()
    pnts = [None] * 4
    lines = [None] * 4
    print("Called generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane")
    variables = {
        "n_turns":n_turns,
        "w":w,
        "h":h,
        "insulation_thickness":insulation_thickness,
        "solh":solh,
        "solw":solw,
        "radius":radius,
        "dr":dr,
        "maxh":maxh,
        "controlledmaxh":controlledmaxh,
        "visu":visu
    }

    for name,value in variables.items():
        print(f"{name} = {value}")

    # Domains
    geo.SetMaterial(1, "Control")
    geo.SetMaterial(2, "Air")
    for i in range(10):
        geo.SetMaterial(3 + i, f"T{i + 1}")

    # Controlled region
    pnts[0], pnts[1], pnts[2], pnts[3] = [(x, y) for x, y in [(0, 0), (0, solh*1.05), (solw*1.05, solh*1.05), (solw*1.05, 0)]]
    p = [geo.AppendPoint(*pnt) for pnt in pnts]
    lines[0] = ["line", p[0], p[1]]
    lines[1] = ["line", p[1], p[2]]
    lines[2] = ["line", p[2], p[3]]
    lines[3] = ["line", p[3], p[0]]
    geo.Append(lines[0], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
    geo.Append(lines[1], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[2], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[3], leftdomain=0, rightdomain=1, bc="bottom", maxh=controlledmaxh)

    # Solenoid
    z0 = float(0)  # Starting position for the turns
    for i in range(n_turns):
        x_min = radius + dr
        x_max = radius + dr + w
        y_min = z0 + insulation_thickness
        y_max = z0 + insulation_thickness + h

        geo.AddRectangle(p1=(x_min, y_min),
                         p2=(x_max, y_max),
                         leftdomain=3 + i,
                         rightdomain=2)
        z0 += (h + insulation_thickness)

    # Simulation area
    pnts.append((0, max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((3 * (radius + dr + w), max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((3 * (radius + dr + w),0))

    p.append(geo.AppendPoint(*pnts[-3]))
    p.append(geo.AppendPoint(*pnts[-2]))
    p.append(geo.AppendPoint(*pnts[-1]))

    geo.Append(["line", p[1], p[-3]], leftdomain=0, rightdomain=2, bc="left")
    geo.Append(["line", p[-3], p[-2]], leftdomain=0, rightdomain=2, bc="top")
    geo.Append(["line", p[-2], p[-1]], leftdomain=0, rightdomain=2, bc="right")
    geo.Append(["line", p[-1], p[3]], leftdomain=0, rightdomain=2, bc="bottom")


    print(geo.GetNPoints())
    print(geo.GetNDomains())
    # Generate meshes
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ngsolve.Mesh(ngmesh)

    return mesh, geo

def generate_ngsolve_geometry_and_mesh_w_ins_offset_geomopt(n_turns=10, wstep=0.0005, h=0.0015, insulation_thickness=0.00006,
                                                    solh=0.05, solw=0.05, minradius = 0.005, maxradius=0.0505,
                                                    maxh=0.001, controlledmaxh=0.001, visu=1):
    geo = SplineGeometry()
    points = {}  # Dictionary to store unique points
    lines = set()  # Set to store unique lines

    def AppendUniquePoint(x, y):
        """Appends a point only if it doesn't already exist."""
        if (x, y) not in points:
            points[(x, y)] = geo.AppendPoint(x, y)
        return points[(x, y)]

    def AppendUniqueLine(p,p1,p2,leftdomain,rightdomain):
        """Appends a line only if it doesn't already exist and ensures correct orientation."""
        if (p1,p2) not in lines and (p2,p1) not in lines:
            geo.Append(["line",p1,p2],leftdomain=leftdomain,rightdomain=rightdomain)
            # print(
            #     f"Appended a line from p[{p.index(p1)}] to p[{p.index(p2)}] "
            #     f"with coordinates ({p1}) -> ({p2}) "
            #     f"with left domain number {leftdomain} and right domain number {rightdomain}"
            # )
            lines.add((p1,p2))
    def AddRectangle2(geo, p1, p2, leftdomain, rightdomain_N, rightdomain_S, rightdomain_W, rightdomain_E):
        """Adds a rectangle with separate materials for each edge, avoiding duplicates and ensuring correct orientation."""
        p = [
            AppendUniquePoint(p1[0], p1[1]),
            AppendUniquePoint(p2[0], p1[1]),
            AppendUniquePoint(p2[0], p2[1]),
            AppendUniquePoint(p1[0], p2[1])
        ]
        AppendUniqueLine(p,p[0], p[1], leftdomain, rightdomain_S)  # Bottom
        AppendUniqueLine(p,p[1], p[2], leftdomain, rightdomain_E)  # Right
        AppendUniqueLine(p,p[2], p[3], leftdomain, rightdomain_N)  # Top
        AppendUniqueLine(p,p[3], p[0], leftdomain, rightdomain_W)  # Left
        print([p,rightdomain_W,leftdomain,rightdomain_E])
    # Domains
    geo.SetMaterial(1, "Control")
    geo.SetMaterial(2, "Air")
    # Controlled region
    pnts = [(0, 0), (0, solh*1.05), (solw*1.05, solh*1.05), (solw*1.05, 0)]
    p = [geo.AppendPoint(*pnt) for pnt in pnts]
    geo.Append(["line", p[3], p[0]], leftdomain=0, rightdomain=1, bc="bottom", maxh=controlledmaxh)
    geo.Append(["line", p[0], p[1]], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
    geo.Append(["line", p[1], p[2]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(["line", p[2], p[3]], leftdomain=2, rightdomain=1, maxh=controlledmaxh)

    material_index = 3
    # Generate solenoid optimization space
    z0 = float(0)  # Starting position for the turns
    for i in range(n_turns):
        x_min = minradius
        for r in np.arange(minradius, maxradius, wstep):
            geo.SetMaterial(material_index,f"T{i + 1}R{round(r,4)}")
            x_max = x_min + wstep  # Fixed width to 0.5 mm
            y_min = z0 + insulation_thickness
            y_max = z0 + insulation_thickness + h
            print(r,x_min,x_max)
            leftdomain = material_index
            if r ==  np.arange(minradius, maxradius, wstep)[0]:
                # print("Start of the line")
                AddRectangle2(
                    geo,
                    p1=(x_min, y_min),
                    p2=(x_max, y_max),
                    leftdomain=leftdomain,
                    rightdomain_N=2,
                    rightdomain_S=2,
                    rightdomain_W=2,
                    rightdomain_E=leftdomain + 1
                )
            elif  r == np.arange(minradius, maxradius, wstep)[-1]:
                # print("End of the line")
                AddRectangle2(
                    geo,
                    p1=(x_min, y_min),
                    p2=(x_max, y_max),
                    leftdomain=leftdomain,
                    rightdomain_N=2,
                    rightdomain_S=2,
                    rightdomain_W=leftdomain - 1,
                    rightdomain_E=2
                )
            else:
                # print("Midpoint")
                AddRectangle2(
                    geo,
                    p1=(x_min, y_min),
                    p2=(x_max, y_max),
                    leftdomain=leftdomain,
                    rightdomain_N=2,
                    rightdomain_S=2,
                    rightdomain_W=leftdomain - 1,
                    rightdomain_E=leftdomain + 1
                )
            x_min=x_max
            material_index += 1

        z0 += (h + insulation_thickness)

    # Simulation area
    pnts.append((0, max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((65e-3, max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((65e-3, 0))

    p.append(AppendUniquePoint(*pnts[-3]))
    p.append(AppendUniquePoint(*pnts[-2]))
    p.append(AppendUniquePoint(*pnts[-1]))

    geo.Append(["line", p[-1], p[3]], leftdomain=0, rightdomain=2, bc="bottom")
    geo.Append(["line", p[1], p[-3]], leftdomain=0, rightdomain=2, bc="left")
    geo.Append(["line", p[-3], p[-2]], leftdomain=0, rightdomain=2, bc="top")
    geo.Append(["line", p[-2], p[-1]], leftdomain=0, rightdomain=2, bc="right")

    # Generate meshes
    # print("Generating mesh..")
    ngmesh = geo.GenerateMesh(maxh=maxh)
    # print("mesh generated, transforming..")
    mesh = ngsolve.Mesh(ngmesh)
    # print("Mesh transformed to ngmesh")

    return mesh, geo

def generate_ngsolve_geometry_and_mesh_w_ins_offset_halfplane_radopt(n_turns=10, w=0.01, h=0.015, insulation_thickness=0.00006,
                                                    solh=0.005, solw=0.005, radius=None, dr=0.5e-3,
                                                    maxh=0.001, controlledmaxh=0.001, visu=1):
    if radius is None or len(radius) != n_turns:
        raise ValueError("The 'radius' parameter must be a list of length equal to 'n_turns'.")

    geo = SplineGeometry()
    pnts = [None] * 4
    lines = [None] * 4

    # Domains
    geo.SetMaterial(1, "Control")
    geo.SetMaterial(2, "Air")
    for i in range(n_turns):
        geo.SetMaterial(3 + i, f"T{i + 1}")

    # Controlled region
    pnts[0], pnts[1], pnts[2], pnts[3] = [(x, y) for x, y in [(0, 0), (0, solh*1.05), (solw*1.05, solh*1.05), (solw*1.05, 0)]]
    p = [geo.AppendPoint(*pnt) for pnt in pnts]
    lines[0] = ["line", p[0], p[1]]
    lines[1] = ["line", p[1], p[2]]
    lines[2] = ["line", p[2], p[3]]
    lines[3] = ["line", p[3], p[0]]
    geo.Append(lines[0], leftdomain=0, rightdomain=1, bc="left", maxh=controlledmaxh)
    geo.Append(lines[1], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[2], leftdomain=2, rightdomain=1, maxh=controlledmaxh)
    geo.Append(lines[3], leftdomain=0, rightdomain=1, bc="bottom", maxh=controlledmaxh)

    # Solenoid
    z0 = float(0)  # Starting position for the turns

    for i in range(n_turns):
        current_radius = radius[i]  # Unique radius for each turn from the input vector
        x_min = current_radius + dr
        x_max = current_radius + dr + w
        y_min = z0 + insulation_thickness
        y_max = z0 + insulation_thickness + h

        geo.AddRectangle(p1=(x_min, y_min),
                         p2=(x_max, y_max),
                         leftdomain=3 + i,
                         rightdomain=2)
        z0 += (h + insulation_thickness)

    # Simulation area
    pnts.append((0, max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((3 * (max(radius) + w), max(1.6 * n_turns * (h + 2 * insulation_thickness), 2 * solh)))
    pnts.append((3 * (max(radius) + w), 0))

    p.append(geo.AppendPoint(*pnts[-3]))
    p.append(geo.AppendPoint(*pnts[-2]))
    p.append(geo.AppendPoint(*pnts[-1]))

    geo.Append(["line", p[1], p[-3]], leftdomain=0, rightdomain=2, bc="left")
    geo.Append(["line", p[-3], p[-2]], leftdomain=0, rightdomain=2, bc="top")
    geo.Append(["line", p[-2], p[-1]], leftdomain=0, rightdomain=2, bc="right")
    geo.Append(["line", p[-1], p[3]], leftdomain=0, rightdomain=2, bc="bottom")

    # Generate meshes
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ngsolve.Mesh(ngmesh)

    return mesh, geo