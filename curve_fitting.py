import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import plotly.graph_objects as go
from scipy.optimize import minimize
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class MaterialThicknessData:
    thickness: float
    displacement: NDArray = np.array([], dtype=np.float64)
    velocity: NDArray = np.array([], dtype=np.float64)
    force: NDArray = np.array([], dtype=np.float64)


def load_material(file_path: Path) -> list[MaterialThicknessData]:
    with open(file_path) as file:
        material_data: list[MaterialThicknessData] = []
        for line in file:
            if not line.strip():  # Skip empty lines
                continue
            data = json.loads(line)
            thickness = data['thickness']
            same_thickness_list = list(filter(lambda x: x.thickness == thickness, material_data))
            assert (len(same_thickness_list) < 2)
            if not same_thickness_list:
                material_data.append(MaterialThicknessData(thickness))
                thickness_data = material_data[-1]
            else:
                thickness_data: MaterialThicknessData = same_thickness_list[0]

            displacements = data["displacements"]
            velocities = data["velocity"] * np.ones(len(displacements))
            forces = data["forces"]
            thickness_data.displacement = np.hstack([thickness_data.displacement, displacements])
            thickness_data.velocity = np.hstack([thickness_data.velocity, velocities])
            thickness_data.force = np.hstack([thickness_data.force, forces])
    return material_data


from scipy.optimize import least_squares


def fit_data(material_data):
    results = []
    thicknesses = []

    for data in material_data:
        thicknesses.append(data.thickness * 1000)

        def residuals(x, data):
            c, k1, k3 = x
            # Return the residual vector (not the norm)
            residual = data.force.copy()
            residual -= c * data.velocity
            residual -= k1 * data.displacement * 1e6
            residual -= k3 * data.displacement ** 3 * 1e9
            return residual

        # Using least_squares - note bounds format is (lower, upper) for each variable
        solution = least_squares(
            residuals,
            np.array([0, 0, 0]),
            bounds=(
                np.array([0.0, 0.0, 0.0]),
                np.array([np.inf, np.inf, np.inf])
            ),  # (lower, upper)
            loss='linear',
            args=(data,),  # Pass additional arguments
            max_nfev=100000,
            method='dogbox'  # Trust Region Reflective - good for bounds
        )

        assert solution.success, f"Optimization failed: {solution.message}"
        results.append(solution.x * np.array([1, 1e6, 1e9]))
    # print(results)
    return np.array(thicknesses), np.array(results)


def main():
    file_path = Path(__file__).parent / "fenicsx_results" / "parametric_results_kevlar_29.jsonl"
    material_data = load_material(file_path)

    results = []
    thicknesses = []
    for data in material_data:
        thicknesses.append(data.thickness * 1000)

        def residuals(x, data):
            c, k1, k3 = x
            # Return the residual vector (not the norm)
            residual = data.force.copy()
            residual -= c * data.velocity
            residual -= k1 * data.displacement * 1e6
            residual -= k3 * data.displacement ** 3 * 1e9
            return residual

        # Using least_squares - note bounds format is (lower, upper) for each variable
        solution = least_squares(
            residuals,
            np.array([0, 0, 0]),
            bounds=(
                np.array([0.0, 0.0, -np.inf]),
                np.array([np.inf, np.inf, np.inf])
            ),  # (lower, upper)
            loss='linear',
            args=(data,),  # Pass additional arguments
            max_nfev=100000,
            method='dogbox'  # Trust Region Reflective - good for bounds
        )
        print(solution)
        assert solution.success, f"Optimization failed: {solution.message}"
        results.append(solution.x)
        c, k1, k3 = solution.x
        k3 *= 1e9
        k1 *= 1e6
        fig, ax = plt.subplots()
        velocity = np.unique(data.velocity)[5]
        mask = data.velocity == velocity
        ax.plot(data.displacement[mask], data.force[mask])
        t = np.linspace(0, max(data.displacement), 10000)

        def force(displacement, velocity):
            return c * velocity + k1 * displacement + k3 * displacement ** 3

        vectorized_force = np.vectorize(force)
        ax.plot(t, vectorized_force(t, velocity * np.ones(len(t))))
        ax.set_title(f"c={c:.3g}   k1={k1:.3g}    k3={k3:.3g}")
        plt.show()
    c, k1, k3 = list(zip(*results))
    print(c)
    print(thicknesses)
    fig, ax = plt.subplots()
    ax.plot(thicknesses, c)
    ax.set_ylabel("Coefficient c (Ns/m)")
    ax.set_xlabel("Thickness (mm)")
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(thicknesses, k1)
    ax.set_ylabel("Coefficient k1 (N/m)")
    ax.set_xlabel("Thickness (mm)")
    plt.show()
    fig, ax = plt.subplots()
    ax.set_ylabel("Coefficient k3 (N/m^3)")
    ax.set_xlabel("Thickness (mm)")
    ax.plot(thicknesses, k3)
    plt.show()
    # velocity = np.unique(material_data.velocity)[6]
    # mask = material_data.velocity == velocity
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(material_data.displacement[mask], material_data.force[mask])
    # plt.show()
    x = material_data[7].displacement
    y = material_data[7].velocity
    z = material_data[7].force

    # Create grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    # Surface plot
    fig = go.Figure(data=[
        go.Surface(x=Xi, y=Yi, z=Zi)
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='Displacement',
            yaxis_title='Velocity',
            zaxis_title='Force'
        )
    )

    fig.show()


if __name__ == '__main__':
    main()
