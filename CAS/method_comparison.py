from FEM_calculation import fem_simulation
from Functions.opt_fun import *
from Functions.ROM import *

n_turns = 10
i_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]
insulation_thickness=0.00006
I = np.ones(10) * 3.18
# radius = [r * 1e-3 for r in [11, 8, 14, 10, 6, 11, 8, 11, 13, 13]]
solh=0.005
solw=0.005
coordinates = generate_coordinates(0.0002,solw,0,solh,11,6)
maxh=0.0005
controlledmaxh=0.0005
B_target=2e-3
B_ref = [B_target for _ in coordinates]

w = 0.001
wstep = 0.0005
h = 0.0015
sigma_coil = 4e7
minradius=0.0055
maxradius=0.035

results = {}
print("---------------------generation---------------------------------------")
#
# generate_reduced_order_model_geomopt(n_turns, w, h, insulation_thickness, solh, solw, maxh, controlledmaxh,
#                                   wstep, minradius, maxradius, filename)

print("---------------------model comparison---------------------------------")

radius = [r * 1e-3 for r in np.random.randint(minradius*1000+1, maxradius*1000-1, size=10)]
# Load ROM
mesh,Vh,rA,CM = load_radopt(filename,n_turns,i_vec,I,radius)

for iteration in range(30):
    radius = [r * 1e-3 for r in np.random.randint(minradius * 1000 + 1,maxradius * 1000 - 1,size=10)]
    print("---------------------")
    print(f"Random radius (mm): {np.array(radius)*1e3}")
    print("---------------------ROM---------------------------------------")



    # Calc with ROM
    Br_vals, Bz_vals, B_vals, f1, f2, f3, timings, _,_,_ = calc_with_ROM(
        mesh, Vh, rA, CM, n_turns, i_vec, I, radius, coordinates, B_ref, w, wstep, h, sigma_coil
    )

    # Dump results
    print("B values:", B_vals)
    print("Metric f1:", f1)
    print("Metric f2:", f2)
    print("Metric f3:", f3)
    print("Timings:", timings)
    # Calc with FEM
    print("---------------------FEM---------------------------------------")
    [Br_valsFEM, Bz_valsFEM, B_valsFEM, f1FEM, f2FEM, f3FEM, timingsFEM,_,_,_,_] =  fem_simulation(n_turns, w, h, radius,
                        insulation_thickness, solh, solw, maxh,
                        controlledmaxh, B_target, I, sigma_coil)
    # Dump results
    print("B values:", B_valsFEM)
    print("Metric f1:", f1FEM)
    print("Metric f2:", f2FEM)
    print("Metric f3:", f3FEM)
    print("Timings:", timingsFEM)

    errors = calculate_errors(B_vals,B_valsFEM)
    print(f"Mean absolute error: {errors[0]} T")
    print(f"Maximum absolute error: {errors[1]} T")
    print(f"Mean relative error: {errors[2]*100} %")
    print(f"Maximum relative error: {errors[3]*100} %")

    results[iteration] = {
        "radius": radius,
        "ROM_B_values": B_vals.tolist(),
        "FEM_B_values": B_valsFEM.tolist(),
        "f1_ROM": f1,
        "f2_ROM": f2,
        "f3_ROM": f3,
        "f1_FEM": f1FEM,
        "f2_FEM": f2FEM,
        "f3_FEM": f3FEM,
        "timingsFEM": timingsFEM,
        "timings": timings,
        "errors": {
            "mean_abs_error": errors[0],
            "max_abs_error": errors[1],
            "mean_rel_error": errors[2] * 100,
            "max_rel_error": errors[3] * 100,
        },
    }

output_filename = "../Solenoid/results1.pkl"
with open(output_filename, "wb") as file:
    pickle.dump(results, file)