import time
import numpy as np
import pandas as pd
from datetime import  datetime
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from FEM_calculation import fem_simulation
from Functions.opt_fun import *
from Functions.ROM import *

# Bemeneti paraméterek

n_turns = 10
i_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]
insulation_thickness = 0.00006
solh = 0.005
solw = 0.005
coordinates = generate_coordinates(0.0002, solw, 0, solh, 11, 6)
maxh = 0.0002
controlledmaxh = 0.0002
B_target = 2e-3
B_ref = [B_target for _ in coordinates]
w = 0.001
wstep = 0.0005
h = 0.0015
sigma_coil = 4e7
minradius = 0.006 # m
maxradius = 0.020  # m
mincurrent = 0 # A
maxcurrent = 5 # A

maxh_str = f"{maxh*1000:.1f}".replace('.', 'p') + "mm"
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # Pl. "20250610_142530"
xlname = f"results_{maxh_str}_{now_str}_Combined.xlsx"
filename = f"results_{maxh_str}_{now_str}_Combined.pkl"

# ROM betöltése vagy generálása
mesh, Vh, rA, CM = generate_reduced_order_model_geomopt(
    n_turns, w, h, insulation_thickness, solh, solw, maxh, controlledmaxh,
    wstep, minradius - 0.0005, maxradius + 0.0005+wstep, filename)

# mesh, Vh, rA, CM = load_radopt(filename, n_turns, i_vec, np.ones(n_turns), [0.01] * n_turns)

n_vars = n_turns * 2

class CoilOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=n_vars,
            n_obj=2,
            n_constr=0,
            xl=np.array([minradius]*n_turns + [mincurrent]*n_turns),
            xu=np.array([maxradius]*n_turns + [maxcurrent]*n_turns),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1_values = []
        f2_values = []
        for i in range(len(x)):
            indiv = x[i]
            radii = np.round(indiv[:n_turns] * 2000) / 2000  # 0.5 mm-es kerekítés
            currents = indiv[n_turns:]

            Br_vals, Bz_vals, B_vals, f1, f2, _, _, _, _, _ = calc_with_ROM(
                mesh, Vh, rA, CM, n_turns, i_vec, currents, radii,
                coordinates, B_ref, w, wstep, h, sigma_coil
            )
            f1_values.append(f1)
            f2_values.append(f2)
        out["F"] = np.column_stack([f1_values, f2_values])

class OptimizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.generations = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F=algorithm.pop.get("F")
        f1_min = F[:, 0].min()
        f2_min = F[:, 1].min()
        X = algorithm.pop.get("X")
        best_index = F[:,0].argmin()
        best_X = X[best_index]
        print(f"Generation {gen}: Best f1 = {f1_min:.6f}, Best f2 = {f2_min:.6f}")
        print(f"             Best X = {best_X}")
        self.generations.append((gen, f1_min, f2_min))

optimStart = time.perf_counter()
problem = CoilOptimizationProblem()
algorithm = NSGA2(pop_size=50)
callback = OptimizationCallback()

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 50),
    seed=1,
    verbose=True,
    callback=callback,
)
optimEnd = time.perf_counter()

pareto_front = res.F
pareto_all = res.X
pareto_radii = np.round(pareto_all[:, :n_turns] * 2000) / 2000
pareto_currents = pareto_all[:, n_turns:]

data = {
    "f1 (Magnetic field error)": pareto_front[:, 0],
    "f2 (Robustness)": pareto_front[:, 1],
    "Radii (mm)": [", ".join(map(lambda r: f"{r*1e3:.3f}", rset)) for rset in pareto_radii],
    "Currents (A)": [", ".join(map(lambda i: f"{i:.3f}", iset)) for iset in pareto_currents],
    "Optimization Time (s)": optimEnd - optimStart,
}

rom_f1_values = []
rom_f2_values = []
fem_f1_values = []
fem_f2_values = []
fem_times = []
rom_times = []
fem_Bz = []
fem_Br = []
rom_Bz = []
rom_Br = []

print("starting post-simulation")
# FEM és ROM számítások minden Pareto pontnál
for ii in range(len(pareto_radii)):
    radii_meters = pareto_radii[ii]
    I = pareto_currents[ii]
    print(radii_meters)
    print(I)
    # FEM számítás időzítéssel
    fem_start = time.process_time()
    Br_valsFEM, Bz_valsFEM, B_valsFEM, f1FEM, f2FEM, _, _, _, _, _, _ = fem_simulation(
        n_turns=n_turns, w=w, h=h, radius=radii_meters, insulation_thickness=insulation_thickness, solh=solh, solw=solw, maxh=maxh, controlledmaxh=controlledmaxh, B_target=B_target, I=I,sigma_coil=sigma_coil
    )
    fem_end = time.process_time()

    # ROM számítás időzítéssel
    rom_start = time.process_time()
    Br_valsROM, Bz_valsROM, B_valsROM, f1ROM, f2ROM, _, _, _, _, _ = calc_with_ROM(
        mesh, Vh, rA, CM, n_turns, i_vec, I, radii_meters, coordinates, B_ref, w, wstep, h, sigma_coil)

    rom_end = time.process_time()

    # Eredmények mentése
    fem_f1_values.append(f1FEM)
    fem_f2_values.append(f2FEM)
    rom_f1_values.append(f1ROM)
    rom_f2_values.append(f2ROM)
    fem_times.append(fem_end - fem_start)
    rom_times.append(rom_end - rom_start)
    fem_Bz.append(Bz_valsFEM)
    fem_Br.append(Br_valsFEM)
    rom_Bz.append(Bz_valsROM)
    rom_Br.append(Br_valsROM)

data["FEM f1"] = fem_f1_values
data["FEM f2"] = fem_f2_values
data["ROM f1"] = rom_f1_values
data["ROM f2"] = rom_f2_values
data["FEM Time (s)"] = fem_times
data["ROM Time (s)"] = rom_times
data["FEM Bz"] = [", ".join(map(str, Bz)) for Bz in fem_Bz]
data["FEM Br"] = [", ".join(map(str, Br)) for Br in fem_Br]
data["ROM Bz"] = [", ".join(map(str, Bz)) for Bz in rom_Bz]
data["ROM Br"] = [", ".join(map(str, Br)) for Br in rom_Br]

results_df = pd.DataFrame(data)

# Excel fájlba mentés
results_df.to_excel(xlname, sheet_name="Pareto Front", index=False)

print("Results saved to optimization_results_with_times_and_Bfields.xlsx")
