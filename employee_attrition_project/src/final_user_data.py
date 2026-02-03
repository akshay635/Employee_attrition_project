# -*- coding: utf-8 -*-

def final_inputs(inputs, medians, modes):
    # baseline input features on which models are trained
    baseline = {}
    
    # using for loops to store medians and modes for unwanted features
    for key, values in medians.items():
        medians[key] = int(values)
        
    for i, j in modes.items():
        modes[i] = str(modes[i])
        
    baseline.update(medians)
    baseline.update(modes)

    # Updating user inputs in the baseline inputs
    final_inputs = baseline.copy()
    final_inputs.update(inputs)
    return final_inputs

