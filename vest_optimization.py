from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from dataclasses import dataclass
from curve_fitting import load_material, fit_data
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


@dataclass(slots=True)
class Coefficients:
    c: float
    k1: float
    k3: float


@dataclass(slots=True)
class Material:
    name: str
    density: float
    cost: callable
    k1: callable
    k3: callable
    c: callable


def create_materials():
    cost_func = lambda rate, density: (lambda x: x * rate * density, density)
    cost_density = {'alumina': cost_func(20, 3700), 'ar500': cost_func(5, 7860),
                    'bor_carbide': cost_func(90, 2510),
                    'kevlar_29': cost_func(160, 1400), 'kevlar_49': cost_func(180, 1467),
                    'sil_carbide': cost_func(30, 3163),
                    'titanium_alloy': cost_func(40, 4428), 'twaron': cost_func(140, 1450)}

    keys = cost_density.keys()
    file_path = Path(__file__).parent / "fenicsx_results"
    materials = []
    for name in keys:
        # if name not in ['alumina', 'kevlar_29']:
        #     continue
        material_path = file_path / f"parametric_results_{name}.jsonl"
        material_data = load_material(material_path)
        thicknesses, results = fit_data(material_data)
        materials.append(Material(name =name, cost=cost_density[name][0], density=cost_density[name][1],
                                  c=interp1d(np.hstack(([0], thicknesses)), np.hstack(([0], results[:, 0])), fill_value='extrapolate'),
                                  k1=interp1d(np.hstack(([0], thicknesses)), np.hstack(([0], results[:, 1])), fill_value='extrapolate'),
                                  k3=interp1d(np.hstack(([0], thicknesses)), np.hstack(([0], results[:, 2])), fill_value='extrapolate')))
    return materials


class Event:
    __slots__ = ['func', 'terminal', 'direction']

    def __init__(self, func, terminal=True, direction=0):
        self.func = func
        self.terminal = terminal
        self.direction = direction

    def __call__(self, t, y):
        return self.func(t, y)


class VestOptimization:
    __slots__ = ['materials', 'initial_velocity', 'bullet_mass', 'mass_limit', 'cost_limit', 'displacement_limit',
                 'total_thickness_limit', 'coefficients', 'densities', 'area', '_max_displacement', '_max_force']

    def __init__(self, materials, initial_velocity, bullet_mass, mass_limit, cost_limit, displacement_limit,
                 total_thickness_limit, area):
        self._max_displacement = Event(self._max_displacement_event, terminal=True, direction=-1)
        self._max_force = Event(self._max_force_event, terminal=False, direction=-1)

        self.materials: list[Material] = materials
        self.densities = np.array([material.density for material in self.materials])
        self.initial_velocity = initial_velocity
        self.bullet_mass = bullet_mass
        self.mass_limit = mass_limit
        self.cost_limit = cost_limit
        self.displacement_limit = displacement_limit
        self.total_thickness_limit = total_thickness_limit

        self.area = area * 2 # Frontal Protection + Back's Protection

        self.coefficients: Coefficients = Coefficients(0, 0, 0)

    def evaluate(self, x):
        # print(x)
        self._update_coefficients(x)
        # print(self.coefficients)
        sol = solve_ivp(self.equations_system(), [0, 1],
                        np.array([0, self.initial_velocity]), method='BDF', dense_output=True,
                        events=[self._max_displacement, self._max_force])
        if sol.status != 1:
            # print("Erro")
            # print(sol)
            # t = np.linspace(0, sol.t[-1], 2000)
            # y = sol.sol(t)
            # fig3, axes = plt.subplots(3, 1)
            # axes[0].plot(t * 1000, y[0, :] * 1000)
            # axes[0].set_ylabel("Position (mm)")
            # axes[0].set_xlabel("Time (ms)")
            #
            # axes[1].plot(t * 1000, y[1, :])
            # axes[1].set_ylabel("Velocity (m/s)")
            # axes[1].set_xlabel("Time (ms)")
            #
            # force = lambda y: (
            #         y[0, :] * self.coefficients.k1
            #         + y[0, :] ** 3 * self.coefficients.k3
            #         + y[1, :] * self.coefficients.c
            # )
            # axes[2].plot(t * 1000, force(y))
            # axes[2].set_ylabel("Force (N)")
            # axes[2].set_xlabel("Time (ms)")
            #
            # fig3.tight_layout()
            # plt.show()

            return 1e20, np.array([1e20 for _ in range(4)])
        candidates = [self.force(0, self.initial_velocity), self.force(sol.y[0][-1], sol.y[1][-1])]
        if sol.y_events[1].size > 0:
            candidates.extend(np.apply_along_axis(lambda x: self.force(*x), 1, sol.y_events[1]))
        objective_value = max(candidates)

        try:
            max_displacement = sol.y_events[0][0][0]
        except IndexError as error:
            print(x)
            print(sol)
            raise error
        # print(max_displacement)
        # Constraints
        constraint_thickness = np.sum(x) - self.total_thickness_limit
        constraint_mass = np.sum(x * self.densities) * self.area / 1000 - self.mass_limit
        constraint_cost = np.sum(
            [material.cost(x[i] * self.area/1000) for i, material in enumerate(self.materials)]) - self.cost_limit
        constraint_displacement = max_displacement * 1000 - self.displacement_limit

        constraints = np.array([constraint_thickness, constraint_mass, constraint_cost, constraint_displacement])

        return objective_value, constraints

    def evaluate_penalized(self, x):
        penalization_factor = 1e6
        objective_value, constraints = self.evaluate(x)
        constraints_conditioning = 1e7 * np.array([1 / 25, 1 / 16, 1 / 1500, 1 / 20])
        return objective_value + np.sum(
            constraints_conditioning * np.clip(constraints, 0, None) ** 2) * penalization_factor

    def equations_system(self):
        def equation1(t, x):
            u = x[0]
            v = x[1]
            du_dt = v
            return du_dt

        def equation2(t, x):
            u = x[0]
            v = x[1]
            dv_dt = self.coefficients.c * v
            dv_dt += self.coefficients.k1 * u
            dv_dt += self.coefficients.k3 * u ** 3
            dv_dt /= -self.bullet_mass
            return dv_dt

        return lambda t, x: [equation1(t, x), equation2(t, x)]

    def _update_coefficients(self, x: np.ndarray) -> None:
        self.coefficients = Coefficients(0, 0, 0)
        for i, material in enumerate(self.materials):
            # thickness_factor = max((1-abs(max((0.05-x[i]),0))),0)
            self.coefficients.k1 += material.k1(x[i])
            self.coefficients.k3 += material.k3(x[i])
            self.coefficients.c += material.c(x[i])

    def _max_displacement_event(self, t, y):
        return y[1]

    def _max_force_event(self, t, y):
        u = y[0]
        v = y[1]
        error = (self.coefficients.k1 + 3 * self.coefficients.k3 * u ** 2) * v * self.bullet_mass
        error -= (
                             self.coefficients.k1 * u + self.coefficients.k3 * u ** 3 + v * self.coefficients.c) * self.coefficients.c
        return error

    def force(self, u, v):
        return u * self.coefficients.k1 + u ** 3 * self.coefficients.k3 + v * self.coefficients.c


def main():
    materials = create_materials()
    print(materials)
    for material in materials:
        print(f"{material.name}, espessura {1}, k1: {material.k1(0)}, k3: {material.k3(0)}, c: {material.c(0)}")
    problem = VestOptimization(materials, initial_velocity=350, bullet_mass=0.009, mass_limit=16, cost_limit=1500,
                               displacement_limit=20, total_thickness_limit=25, area=0.25)

    print(problem.evaluate(np.array([25 * 1 / len(materials) for _ in range(len(materials))])))


if __name__ == '__main__':
    main()
