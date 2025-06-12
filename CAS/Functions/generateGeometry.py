from ngsolve import *
from netgen.geom2d import SplineGeometry
import ngsolve
n_turns = 10
w = .01
h = .015
radius = .1
gap = .00
maxh = .01
controlledmaxh=.01
def create_solenoid_mesh_direct(n_turns,w,h,radius,gap,maxh,controlledmaxh):
    """
    """
    geo = SplineGeometry()
    geo.AddRectangle(p1=(0,0),
                     p2=(2*radius,2*(n_turns*(h+gap))),
                     bc="outer",
                     leftdomain=1,
                     rightdomain=0)
    geo.AddRectangle(p1=(0,0),
                     p2=(0.05,0.05),
                     bc="inner",
                     leftdomain=2,
                     rightdomain=1)
    # z0 = float(0);  # -(h + gap) * n_turns / 2  # Starting position for the turns
    # for i in range(n_turns):
    #     # print(z0)
    #     x_min = radius + i * .000
    #     x_max = radius + w + i * .000
    #     y_min = z0
    #     y_max = z0 + h
    #     # geo.AddRectangle(p1=(x_min,y_min),
    #     #                  p2=(x_max,y_max),
    #     #                  leftdomain=i+3,
    #     #                  rightdomain=1,
    #     #                  mat=f"turn_{i}")
    #     z0 += gap + h
    geo.SetMaterial(1,"air")
    geo.SetMaterial(2,"air2")
    geo.SetDomainMaxH(2,controlledmaxh)
    ngmesh = geo.GenerateMesh(maxh=maxh)
    # Convert Netgen mesh to NGSolve mesh
    # mesh = ngsolve.Mesh(ngmesh)
    return ngmesh

# import netgen.gui
from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import matplotlib.pyplot as plt
n_turns = 10
# shift = np.zeros((10, 1))
w = .01
h = .015
solh = 0.05
solw = 0.05
radius = .1
maxh = 0.01
controlledmaxh=.001
def generate_geometry_and_plot(n_turns=10, w=0.01, h=0.015, solh=0.05, solw=0.05, radius=0.1, maxh=0.001, controlledmaxh=0.001):
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
    geo.Append (lines[0], leftdomain=0, rightdomain=1, bc="left")
    geo.Append (lines[1], leftdomain=2, rightdomain=1)
    geo.Append (lines[2], leftdomain=2, rightdomain=1)
    geo.Append (lines[3], leftdomain=2, rightdomain=1)


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
    ngmesh = geo.GenerateMesh()

    return ngmesh







