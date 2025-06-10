from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
import numpy as np
from opti.utils import *
from rom.utils import *
class ROMOptimizationProblem(Problem):
    def __init__(self, mode, rom_interface, geometry, param):
        self.mode = mode
        self.rom = rom_interface
        self.n_turns = geometry.n_turns
        self.coordinates = param["evaluation_points"]
        self.B_ref = param.get("B_ref",0.02)
        self.fixed_current = param.get("fixed_current", None)
        self.fixed_radius = param.get("fixed_radius", None)
        self.w = geometry.w
        self.h = geometry.h
        self.wstep = geometry.wstep
        self.sigma_coil = param.get("sigma_coil",4e7)
        self.objectives = param.get("objectives",2)
        min_current = param.get("min_current",0)
        max_current = param.get("max_current",10)

        if mode == "radius":
            n_var = self.n_turns
            xl = np.full(n_var, geometry.minradius+0.0005) # 0.0005 b.c. f2 will calculate a negative offset of 0.0005
            xu = np.full(n_var, geometry.maxradius-geometry.wstep-0.0005) #wstep b.c. inner radius; 0.0005 b.c. f2 will calculate a positive offset of 0.0005
        elif mode == "current":
            n_var = self.n_turns
            xl = np.full(n_var, min_current)
            xu = np.full(n_var, max_current)
        elif mode == "combined":
            n_var = self.n_turns * 2
            xl = np.concatenate([
                np.full(self.n_turns, geometry.minradius+0.0005),
                np.full(self.n_turns, min_current)
            ])
            xu = np.concatenate([
                np.full(self.n_turns, geometry.maxradius-geometry.wstep-0.0005),
                np.full(self.n_turns, max_current)
            ])
        else:
            raise ValueError("Unsupported optimization mode.")

        super().__init__(n_var=n_var, n_obj=self.objectives, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        if self.mode == "radius":
            if self.objectives==3:
                self._evaluate_radius_triobjective(X, out)
            else:
                self._evaluate_radius_biobjective(X,out)
        elif self.mode == "current":
            if self.objectives==3:
                self._evaluate_current_triobjective(X, out)
            else:
                self._evaluate_current_biobjective(X,out)
        elif self.mode == "combined":
            if self.objectives ==3:
                self._evaluate_combined_triobjective(X, out)
            else:
                self._evaluate_combined_biobjective(X,out)
        else:
            raise RuntimeError("Unsupported optimization mode during evaluation.")

    def _evaluate_radius_triobjective(self, X, out):
        f1_list, f2_list, f3_list = [], [], []
        current = self.fixed_current
        for radius in X:
            radius = np.round(radius, 5)
            self.rom.calc_a_with_ROM(radius, current)
            [Bx,By] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius+0.0005,current)
            [Bxp,Byp] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx)**2 + np.array(By)**2)
            Bp_vals = np.sqrt(np.array(Bxp)**2 + np.array(Byp)**2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            f3_list.append(calculate_f3(radius,self.w,self.h,self.sigma_coil,current))
            out["F"] = np.column_stack([f1_list, f2_list, f3_list])
    def _evaluate_radius_biobjective(self, X, out):
        f1_list, f2_list = [], []
        current = self.fixed_current
        for radius in X:
            radius = np.round(radius, 5)
            self.rom.calc_a_with_ROM(radius, current)
            [Bx,By] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius+0.0005,current)
            [Bxp,Byp] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx)**2 + np.array(By)**2)
            Bp_vals = np.sqrt(np.array(Bxp)**2 + np.array(Byp)**2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            out["F"] = np.column_stack([f1_list, f2_list])

    def _evaluate_current_triobjective(self, X, out):
        f1_list, f2_list, f3_list = [], [], []
        radius = self.fixed_radius
        for current in X:
            self.rom.calc_a_with_ROM(radius,current)
            [Bx,By] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius + 0.0005,current)
            [Bxp,Byp] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx) ** 2 + np.array(By) ** 2)
            Bp_vals = np.sqrt(np.array(Bxp) ** 2 + np.array(Byp) ** 2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            f3_list.append(calculate_f3(radius,self.w,self.h,self.sigma_coil,current))
            out["F"] = np.column_stack([f1_list,f2_list,f3_list])

    def _evaluate_current_biobjective(self, X, out):
        f1_list, f2_list = [], []
        radius = self.fixed_radius
        for current in X:
            self.rom.calc_a_with_ROM(radius,current)
            [Bx,By] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius + 0.0005,current)
            [Bxp,Byp] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx) ** 2 + np.array(By) ** 2)
            Bp_vals = np.sqrt(np.array(Bxp) ** 2 + np.array(Byp) ** 2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            out["F"] = np.column_stack([f1_list,f2_list])

    def _evaluate_combined_triobjective(self, X, out):
        f1_list, f2_list, f3_list = [], [], []
        for x in X:
            radius = np.round(x[:self.n_turns], 5)
            current = x[self.n_turns:]
            self.rom.calc_a_with_ROM(radius,current)
            [Bx,By] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius + 0.0005,current)
            [Bxp,Byp] = compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx) ** 2 + np.array(By) ** 2)
            Bp_vals = np.sqrt(np.array(Bxp) ** 2 + np.array(Byp) ** 2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            f3_list.append(calculate_f3(radius,self.w,self.h,self.sigma_coil,current))
            out["F"] = np.column_stack([f1_list,f2_list,f3_list])

    def _evaluate_combined_biobjective(self, X, out):
        f1_list, f2_list = [], []
        for x in X:
            radius = round_to_step(x[:self.n_turns], self.wstep)
            current = x[self.n_turns:]
            self.rom.calc_a_with_ROM(radius,current)
            [Bx,By] = self.rom.compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius + 0.0005,current)
            [Bxp,Byp] = self.rom.compute_B_fields_with_ROM(self.coordinates)
            self.rom.calc_a_with_ROM(radius - 0.0005,current)
            [Bxn,Byn] = self.rom.compute_B_fields_with_ROM(self.coordinates)
            B_vals = np.sqrt(np.array(Bx) ** 2 + np.array(By) ** 2)
            Bp_vals = np.sqrt(np.array(Bxp) ** 2 + np.array(Byp) ** 2)
            Bn_vals = np.sqrt(np.array(Bxn) ** 2 + np.array(Byn) ** 2)
            f1_list.append(calculate_f1(B_vals,self.B_ref))
            f2_list.append(calculate_f2(self.B_ref,Bp_vals,Bn_vals))
            out["F"] = np.column_stack([f1_list,f2_list])

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