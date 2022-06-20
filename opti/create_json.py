"""
Create json file for running hypermapper
"""



import json

scenario = {}
scenario["application_name"] = "raycing"
scenario["optimization_objectives"] = ["FWHM", "t_dist"]#, "Gap"]

scenario["optimization_iterations"] = 150
scenario["normalize_inputs"]=True

scenario["models"] = {}
scenario["models"]["model"] = "gaussian_process"

scenario["input_parameters"] = {}
p = {}
p["parameter_type"] = "real"
p["values"] = [-0.003, 0.003]

y = {}
y["parameter_type"] = "real"
y["values"] = [-0.001, 0.001]

r = {}
r["parameter_type"] = "real"
r["values"] = [-0.001, 0.001]

l = {}
l["parameter_type"] = "real"
l["values"] = [-1.0, 1.0]

v = {}
v["parameter_type"] = "real"
v["values"] = [-2.5, 2.5]

scenario["input_parameters"]["pitch"] = p
scenario["input_parameters"]["yaw"] = y
scenario["input_parameters"]["roll"] = r
scenario["input_parameters"]["lateral"] = l
scenario["input_parameters"]["vertical"] = v

with open("opti\\veritas_raycing_m4_scenario.json", "w") as scenario_file:
    json.dump(scenario, scenario_file, indent=4)