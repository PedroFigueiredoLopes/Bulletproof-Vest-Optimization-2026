import create_mesh
from materials_lib import *
from simulation import SimulationParameters, run_simulation

import copy
import numpy as np
from pathlib import Path
import json

base_parameters = SimulationParameters(thickness=0.01,
                                       width=0.4,
                                       height=0.4,
                                       deg_u=2,
                                       deg_quad=4,
                                       indenter_radius=0.009/2,
                                       velocity = 350,
                                       prescribed_displacement = 0.02,
                                       time_steps = 100,
                                       should_output=False)


def load_data(results_path: Path)-> set:
    # Load existing
    if not results_path.exists():
        return set()
    
    completed = set()
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                completed.add((data['thickness'], data['velocity']))
    return completed

def main():
    material = Titanium_alloy
    
    thicknesses = np.linspace(0.001, 0.016, 10)
    velocities = np.linspace(0.1, 350, 20)
    
    results_path = Path(__file__).parent / f"parametric_results_{material.name}.jsonl"
    completed = load_data(results_path)

    print(f"Already completed: {len(completed)} simulations")
    print(f"Total to run: {len(thicknesses) * len(velocities)}")

    for thickness in thicknesses:
        parameters = copy.deepcopy(base_parameters)
        parameters.thickness = thickness
        domain = create_mesh.create_mesh(mesh_size=0.03,thickness = parameters.thickness, height=parameters.height, width=parameters.width)
        for velocity in velocities:
            if (thickness, velocity) in completed:
                continue  # Skip already done
            parameters.velocity = velocity

            print(f"Running thickness={thickness}, velocity={velocity}")
            displacements, forces = run_simulation(parameters=parameters,domain=domain,material=material)
            with open(results_path, 'a') as f:
                json.dump({
                    'thickness': thickness,
                    'velocity': velocity,
                    'displacements': [float(displacement) for displacement in displacements],  # list, not np array
                    'forces': [float(force) for force in forces],  # Convert if needed
                }, f)
                f.write('\n')

if __name__ == '__main__':
    main()