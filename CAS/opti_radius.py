from datetime import datetime
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from FEM_calculation import fem_simulation
from Functions.opt_fun import *
from Functions.ROM import *

# Bemeneti paraméterek
filename = "../Solenoid/ROM_05mmMesh.pkl"

n_turns = 10
i_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]
insulation_thickness = 0.00006
I = np.ones(10) * 3.18
solh = 0.005
solw = 0.005
coordinates = generate_coordinates(0.0002, solw, 0, solh, 11, 6)
maxh = 0.0005
controlledmaxh = 0.0005
B_target = 2e-3
B_ref = [B_target for _ in coordinates]
w = 0.001
wstep = 0.0005
h = 0.0015
sigma_coil = 4e7
minradius = 6  # Minimális sugár integer érték (mm-ben)
maxradius = 50  # Maximális sugár integer érték (mm-ben)

maxh_str = f"{maxh*1000:.1f}".replace('.', 'p') + "mm"
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # Pl. "20250610_142530"
xlname = f"results_{maxh_str}_{now_str}_Radius.xlsx"

mesh, Vh, rA, CM =generate_reduced_order_model_geomopt(n_turns, w, h, insulation_thickness, solh, solw, maxh, controlledmaxh,
                                  wstep, (minradius-0.5)*1e-3, (maxradius+0.5)*1e-3+wstep, filename)

# ROM előkészítése
mesh, Vh, rA, CM = load_radopt(filename,n_turns,i_vec,I,[0.01] * n_turns)

# Optimalizálási probléma osztálya
class CoilOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_turns, n_obj=2, n_constr=0, xl=minradius, xu=maxradius)

    def _evaluate(self, x, out, *args, **kwargs):
        f1_values = []
        f2_values = []
        # Sugár integer értékre kerekítése
        x_rounded = np.round(x).astype(int)
        for radii in x_rounded:
            radii_meters = radii * 1e-3  # mm -> m átváltás
            Br_vals, Bz_vals, B_vals, f1, f2, _, _, _, _, _ = calc_with_ROM(
                mesh, Vh, rA, CM, n_turns, i_vec, I, radii_meters, coordinates, B_ref, w, wstep, h, sigma_coil
            )
            f1_values.append(f1)
            f2_values.append(f2)
        out["F"] = np.column_stack([f1_values, f2_values])

# Egyéni callback osztály
class OptimizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.generations = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        f1_min = algorithm.pop.get("F")[:, 0].min()
        f2_min = algorithm.pop.get("F")[:, 1].min()
        print(f"Generation {gen}: Best f1 = {f1_min:.6f}, Best f2 = {f2_min:.6f}")
        self.generations.append((gen, f1_min, f2_min))

# Optimalizálás indítása
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

# Optimalizációs idő és Pareto-front eredmények
pareto_front = res.F
pareto_radii = np.round(res.X).astype(int)  # Sugár integer formában

# Pareto-front és optimalizálási idő mentése Excel-be
data = {
    "f1 (Magnetic field error)": pareto_front[:, 0],
    "f2 (Robustness)": pareto_front[:, 1],
    "Radii (mm)": [", ".join(map(str, radii)) for radii in pareto_radii],
    "Optimization Time (s)": optimEnd - optimStart
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

# FEM és ROM számítások minden Pareto pontnál
for radii in pareto_radii:
    radii_meters = radii * 1e-3  # mm -> m átváltás

    # FEM számítás időzítéssel
    fem_start = time.process_time()
    Br_valsFEM, Bz_valsFEM, B_valsFEM, f1FEM, f2FEM, _, _, _, _, _, _ = fem_simulation(
        n_turns, w, h, radii_meters, insulation_thickness, solh, solw, maxh, controlledmaxh, B_target, I, sigma_coil
    )
    fem_end = time.process_time()

    # ROM számítás időzítéssel
    rom_start = time.process_time()
    Br_valsROM, Bz_valsROM, B_valsROM, f1ROM, f2ROM, _, _, _, _, _ = calc_with_ROM(
        mesh, Vh, rA, CM, n_turns, i_vec, I, radii_meters, coordinates, B_ref, w, wstep, h, sigma_coil
    )
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
