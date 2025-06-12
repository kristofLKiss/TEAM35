import time
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from datetime import datetime
from pymoo.core.callback import Callback
from FEM_calculation import fem_simulation_sourceopt
from Functions.opt_fun import *
from Functions.ROM import *

# Bemeneti paraméterek
filename = "../Solenoid/ROM_source_02mmMesh.pkl"

n_turns = 10
insulation_thickness = 0.00006
I = np.ones(10) * 3.18
solh = 0.005
solw = 0.005
coordinates = generate_coordinates(0.0002, solw, 0, solh, 11, 6)
maxh = 0.0002
controlledmaxh = 0.0002
B_target = 2e-3
B_ref = [B_target for _ in coordinates]
w = 0.001
h = 0.0015
sigma_coil = 4e7
mincurrent = 0
maxcurrent = 15
radius = 10e-3

maxh_str = f"{maxh*1000:.1f}".replace('.', 'p') + "mm"
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # Pl. "20250610_142530"
xlname = f"results_{maxh_str}_{now_str}_CurrentTriobj.xlsx"

generate_reduced_order_model_sourceopt(n_turns, w, h, insulation_thickness, solh, solw, radius,
                                               maxh, controlledmaxh, filename)
# ROM előkészítése
[RomAred, RomAredPlus, RomAredMinus, FEspace_red, FEspace_redPlus, FEspace_redMinus, mesh, mesh_minus, mesh_plus] = load_sourceopt(filename)

# Optimalizálási probléma osztálya
class CoilOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_turns, n_obj=3, n_constr=0, xl=mincurrent, xu=maxcurrent)

    def _evaluate(self, x, out, *args, **kwargs):
        f1_values = []
        f2_values = []
        f3_values = []
        for I in x:
            [_, _, _, f1, f2, f3, _, _, _, _] = calc_with_ROM_source_optimization(mesh, mesh_plus, mesh_minus, RomAred, RomAredPlus, RomAredMinus, FEspace_red, FEspace_redPlus, FEspace_redMinus, I, radius, coordinates, B_ref,n_turns,w,h,sigma_coil)
            f1_values.append(f1)
            f2_values.append(f2)
            f3_values.append(f3)
        out["F"] = np.column_stack([f1_values, f2_values, f3_values])

# Egyéni callback osztály
class OptimizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.generations = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        f1_min = F[:, 0].min()
        f2_min = F[:, 1].min()
        f3_min = F[:, 2].min()
        X = algorithm.pop.get("X")
        best_index = F[:,0].argmin()
        best_X = X[best_index]
        print(f"Generation {gen}: Best f1 = {f1_min:.6f}, Best f2 = {f2_min:.6f}, Best f3 = {f3_min:.6f}")
        print(f"             Best X = {best_X}")
        self.generations.append((gen, f1_min, f2_min,f3_min))

# Optimalizálás indítása
optimStart = time.perf_counter()
problem = CoilOptimizationProblem()
algorithm = NSGA2(pop_size=50)
callback = OptimizationCallback()

res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 100),
    seed=1,
    verbose=True,
    callback=callback,
)
optimEnd = time.perf_counter()

# Optimalizációs idő és Pareto-front eredmények
pareto_front = res.F
print(res.X)
pareto_currents = res.X.copy()  # Áram

# Pareto-front és optimalizálási idő mentése Excel-be
data = {
    "f1 (Magnetic field error)": pareto_front[:, 0],
    "f2 (Robustness)": pareto_front[:, 1],
    "f3 (Loss)": pareto_front[:,2],
    "Currents (A)": [", ".join(map(str, currents)) for currents in pareto_currents],
    "Optimization Time (s)": optimEnd - optimStart
}
rom_f1_values = []
rom_f2_values = []
rom_f3_values = []
fem_f1_values = []
fem_f2_values = []
fem_f3_values = []
fem_times = []
rom_times = []
fem_Bz = []
fem_Br = []
rom_Bz = []
rom_Br = []

# FEM és ROM számítások minden Pareto pontnál
for I in pareto_currents:
    print(I)
    # FEM számítás időzítéssel
    fem_start = time.process_time()
    Br_valsFEM, Bz_valsFEM, B_valsFEM, f1FEM, f2FEM, f3FEM, _, _, _, _, _ = fem_simulation_sourceopt(
        n_turns, w, h, None, insulation_thickness, solh, solw, maxh, controlledmaxh, B_target, I, sigma_coil
    )
    fem_end = time.process_time()

    # ROM számítás időzítéssel
    rom_start = time.process_time()
    Br_valsROM, Bz_valsROM, B_valsROM, f1ROM, f2ROM, f3ROM, _, _, _, _ = calc_with_ROM_source_optimization(mesh, mesh_plus, mesh_minus, RomAred, RomAredPlus, RomAredMinus, FEspace_red, FEspace_redPlus, FEspace_redMinus, I, radius, coordinates, B_ref,n_turns,w,h,sigma_coil)
    rom_end = time.process_time()

    # Eredmények mentése
    fem_f1_values.append(f1FEM)
    fem_f2_values.append(f2FEM)
    fem_f3_values.append(f3FEM)
    rom_f1_values.append(f1ROM)
    rom_f2_values.append(f2ROM)
    rom_f3_values.append(f3ROM)
    fem_times.append(fem_end - fem_start)
    rom_times.append(rom_end - rom_start)
    fem_Bz.append(Bz_valsFEM)
    fem_Br.append(Br_valsFEM)
    rom_Bz.append(Bz_valsROM)
    rom_Br.append(Br_valsROM)

data["FEM f1"] = fem_f1_values
data["FEM f2"] = fem_f2_values
data["FEM f3"] = fem_f3_values
data["ROM f1"] = rom_f1_values
data["ROM f2"] = rom_f2_values
data["ROM f3"] = rom_f3_values
data["FEM Time (s)"] = fem_times
data["ROM Time (s)"] = rom_times
data["FEM Bz"] = [", ".join(map(str, Bz)) for Bz in fem_Bz]
data["FEM Br"] = [", ".join(map(str, Br)) for Br in fem_Br]
data["ROM Bz"] = [", ".join(map(str, Bz)) for Bz in rom_Bz]
data["ROM Br"] = [", ".join(map(str, Br)) for Br in rom_Br]

results_df = pd.DataFrame(data)

# Excel fájlba mentés
results_df.to_excel(xlname, sheet_name="Pareto Front", index=False)

print("Results saved")
