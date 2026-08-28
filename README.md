# Bulletproof Vest Optimization
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

> **Academic Project Disclaimer**  
> This repository contains the **bulletproof vest optimisation component** developed as part of a class project on nonlinear engineering optimisation (Otimização Não-linear em Engenharia, Universidade de Aveiro, 2026).
> The implementation is **not validated for real-world ballistic protection design**. It uses a highly simplified 1D Duffing oscillator model fitted to elastic FEniCSx simulations that do not include plasticity, fracture, or material failure. The results should be interpreted as a methodological demonstration, not as actual armour design recommendations.

> Note:
> The version of the this repository is not the latest used in the report (notably missing the black hole algorithm module, which can be found in the brachytherapy reporsitory) which was unfortunately lost. 

## The Full Project Report

The complete project report contains the mathematical formulation, sensitivity analysis, benchmark tests (13-bar truss, analytical functions), and a detailed discussion of results for **both** the bulletproof vest problem and the brachytherapy problem.

The full report (Metaheuristic Optimisation of Nonlinear Engineering Problems) is part of a larger compendium which can be found [here]().

*This repository hosts only the code for the bulletproof vest module. The brachytherapy module is available in a [separate repository](https://github.com/PedroFigueiredoLopes/Brachytherapy-Optimization-2026).*

## Repository Structure

### Optimisation & Physics

| File | Description |
| :--- | :--- |
| `vest_optimization.py` | Core class: defines the Duffing oscillator model, constraints (mass, cost, thickness, displacement), and objective function evaluation. |
| `cma_es.py` | Standalone implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). |
| `main.py` | Main orchestration script. Runs multistart CMA-ES optimisation with configurable parameters. |
| `curve_fitting.py` | Fits Duffing coefficients (c, k₁, k₃) to FEniCSx simulation data using least-squares optimisation. |

### FEniCSx Simulation (Data Generation)

| File | Description |
| :--- | :--- |
| `fenicsx-simulation/simulation.py` | Core FEniCSx simulation runner: sets up the dynamic nonlinear plate problem with Newmark time integration. |
| `fenicsx-simulation/create_mesh.py` | Generates structured hexahedral meshes for the plate geometry using Gmsh. |
| `fenicsx-simulation/materials.py` / `materials_lib.py` | Material property definitions (isotropic and orthotropic). |
| `fenicsx-simulation/constitutive_laws.py` | Implements Hencky (logarithmic) strain with JAX for automatic differentiation. |
| `fenicsx-simulation/boundary_conditions.py` | Clamped boundary conditions and circular indenter region. |
| `fenicsx-simulation/solvers.py` | Custom Newton solver with line search and residual monitoring. |
| `fenicsx-simulation/postprocessing.py` | Von Mises stress projection and visualisation tools. |
| `fenicsx-simulation/main.py` | Batch runner: generates parametric results for multiple thicknesses and impact velocities. |

### Data

| Folder | Description |
| :--- | :--- |
| `fenicsx_results/` | Contains `.jsonl` files with simulation results (force-displacement curves) for each material at various thicknesses and velocities. |
| `fenicsx_results/parametric_results_*.jsonl` | Raw simulation data used for Duffing coefficient fitting. |


## Setup

### Option 1: Optimisation Only (Recommended for most users)

If you only want to run the optimisation (using precomputed coefficients), set up the Python environment:

```bash
uv sync
```

This uses [`uv`](https://docs.astral.sh/uv/) and installs `matplotlib`, `scipy`, `numpy`, `pandas`, `pathos`, `plotly` and `pyvista` as defined in `pyproject.toml`.

*Alternatively, if you prefer `pip`*:
```bash
pip install matplotlib numpy pandas scipy pathos plotly pyvista
```

### Option 2: Full FEniCSx Simulation Suite (Advanced)

To regenerate the material response data from scratch follow [FEniCSx docs](https://docs.fenicsproject.org/) for installation. Make sure to also install JAX see [JAX docs](https://docs.jax.dev/en/latest/installation.html).

> **Note:** The FEniCSx simulation code uses JAX for automatic differentiation of the constitutive laws. This makes the simulation faster but adds installation complexity. The precomputed data in `fenicsx_results/` allows you to run the optimisation **without** installing FEniCSx.

## Data Preparation

The optimisation relies on precomputed Duffing coefficients derived from FEniCSx simulations. The simulation data (`fenicsx_results/*.jsonl`) is included for reference.

### To generate your own coefficients (optional):

1. **Add the material:**
   In the file `materials_lib.py` add your material and properties, then add the cost per kilogram to the dictionary `cost_density` in the `vest_optimization.py` file, following the pattern: `name: cost_func(cost, density)`. Make sure the name and density match that of the material in `materials_lib.py`.

2. **Run the FEniCSx simulations:**
   ```bash
   cd fenicsx-simulation
   python main.py
   ```
   This generates `.jsonl` files for each material. These results should then be moved to the `fenicsx_results/` directory.

3. **(Optional) Customize the fitting of the Duffing coefficients:**
   To customize how simulation data is fitted change the function `fit_data` in the file `curve_fitting.py`.

## Running the Optimisation

### Customisation Options

- **Changing the number of starts:** Modify `N_STARTS` at the top of `main.py`. Each start uses a different random seed.
- **Adjusting the objective:** The default objective minimises peak transmitted force while satisfying constraints on thickness, mass, cost, and displacement. This is defined in `vest_optimization.py`.

### Expected outputs:

- Console logs showing convergence progress for each multistart run.
- A plot showing the best solution's displacement, velocity, and force over time.
- The result of the best run.

> **Performance note:** You can change the number of processes that run in parallel can be changed by changing the line `with Pool(processes=10) as pool:` in `main.py`.

## Development Notes & Transparency

This project was developed within a constrained academic timeline. The following notes provide context for reviewers:

- **FEniCSx simulation limitations:** The FEniCSx simulations are purely elastic (no plasticity, fracture, or material failure). They also do not model interlayer interactions. The simulations were run without exploiting symmetry, which would improve performance. The resulting Duffing coefficients are therefore **not physically realistic for ballistic impact** but serve as a demonstration of the optimisation workflow.
- **Model simplification:** The 1D Duffing oscillator is a strong simplification of the actual multilayer ballistic response. It captures only the essential nonlinear stiffening behaviour and ignores many important physical phenomena.
- **Auxiliary AI assistance:** AI was used throughout the development.
- **Performance:** The codebase prioritises **clarity and modularity** over performance optimisation. It is intended to demonstrate the implementation of metaheuristic algorithms and physics-based modelling, not to serve as a production-grade optimization pipeline.

## Acknowledgments

- **Collaborator:** João Gonçalo Pereira Lopes - contributions to development, algorithm testing, mathematical modelling, and discussions.
- **Python Ecosystem:** Built with NumPy, SciPy, Matplotlib, FEniCSx, JAX, Plotly, Pathos, PyVista  and the broader open-source community.
- **FEniCSx Tutorials:** Adapted code snippets were used in the FEniCSx simulation framework. See [official docs](https://jsdokken.com/dolfinx-tutorial/), [dolfinx_materials](https://github.com/bleyerj/dolfinx_materials) and [Numerical Tours of Computational Mechanics with FEniCSx](https://bleyerj.github.io/comet-fenicsx/index.html).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
