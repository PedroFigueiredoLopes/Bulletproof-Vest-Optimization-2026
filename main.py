from dataclasses import dataclass

import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import freeze_support
import cma_es
from vest_optimization import create_materials, VestOptimization
from copy import deepcopy

N_STARTS = 10  # Number of parallel starts; tune to your CPU count


def run_single_start(args):
    """
    Worker function executed in a separate process.
    Each call runs one full CMA-ES optimization from a randomized starting point.
    Returns (best_result, best_value, history).
    """
    run_id, seed, materials, problem = args

    n = len(materials)
    rng = np.random.default_rng(seed)

    # Randomize the initial mean around the uniform distribution with some noise
    base_thickness = 25.0 / n
    initial_mean = rng.uniform(0.1 * base_thickness, 3 * base_thickness, size=n)
    initial_mean = np.clip(initial_mean, 0.0, None)

    bounds = [(0, np.inf)] * n

    try:
        result, history = cma_es.cma_es(
            objective_function=problem.evaluate_penalized,
            generation_callback=None,
            bounds=bounds,
            scaling=None,
            initial_mean=initial_mean,
            initial_step_size=rng.uniform(0.3, 1.0),  # also randomised
            population_size=None,
            max_iterations=5000,
            seed=int(seed),
            penalty_factor_bounds=1e6,
            return_mean=True,
        )
        value = problem.evaluate_penalized(result)
        print(f"[Start {run_id}] finished — penalised objective: {value:.6f}")
        return result, value, history

    except Exception as exc:
        print(f"[Start {run_id}] failed with exception: {exc}")
        return None, np.inf, {}


def main():
    # ------------------------------------------------------------------ #
    # Multistart parallel section
    # ------------------------------------------------------------------ #
    master_rng = np.random.default_rng()
    seeds = master_rng.integers(0, 2 ** 31 - 1, size=N_STARTS)

    materials = create_materials()
    problem = VestOptimization(
        materials,
        initial_velocity=350,
        bullet_mass=0.009,
        mass_limit=16,
        cost_limit=1500,
        displacement_limit=20,
        total_thickness_limit=25,
        area=0.16,
    )
    args = [(i, int(s), deepcopy(materials), deepcopy(problem)) for i, s in enumerate(seeds)]

    print(f"Launching {N_STARTS} parallel CMA-ES runs …")
    with Pool(processes=10) as pool:
        outcomes = pool.map(run_single_start, args)

    # Filter out failed runs and pick the best
    valid = [(res, val, hist) for res, val, hist in outcomes if res is not None]
    if not valid:
        raise RuntimeError("All multistart runs failed.")

    best_result, best_value, best_history = min(valid, key=lambda x: x[1])

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    print("\n=== Best result across all starts ===")
    for i, thickness in enumerate(best_result):
        print(f"Thickness {thickness}: {materials[i].name}")
    print(f"Total thickness: {np.sum(list(best_result))}mm")
    print(f"Total mass: {np.sum(best_result * problem.densities) * problem.area / 1000:} kg")
    print(f"Total cost: {np.sum([material.cost(best_result[i] * problem.area / 1000) for i, material in enumerate(problem.materials)]):}€")
    print("Generations recorded:", len(best_history.get("mean", [])))
    print("Evaluate (unpenalised):", problem.evaluate(best_result))

    # ------------------------------------------------------------------ #
    # Post-processing plot
    # ------------------------------------------------------------------ #
    problem._update_coefficients(best_result)
    sol = solve_ivp(
        problem.equations_system(),
        [0, 1e-3],
        np.array([0, problem.initial_velocity]),
        dense_output=True,
        events=problem._max_displacement,
    )
    t = np.linspace(0, sol.t[-1], 2000)
    y = sol.sol(t)

    fig3, axes = plt.subplots(3, 1)
    axes[0].plot(t * 1000, y[0, :] * 1000)
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_xlabel("Time (ms)")

    axes[1].plot(t * 1000, y[1, :])
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("Time (ms)")

    force = lambda y: (
            y[0, :] * problem.coefficients.k1
            + y[0, :] ** 3 * problem.coefficients.k3
            + y[1, :] * problem.coefficients.c
    )
    axes[2].plot(t * 1000, force(y))
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Time (ms)")

    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
