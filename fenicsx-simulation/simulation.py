import create_mesh, boundary_conditions
from postprocessing import *
from constitutive_laws import *
from materials import IsotropicMaterial, OrthotropicMaterial, MaterialType

import numpy as np
from dataclasses import dataclass


from mpi4py import MPI
from dolfinx import mesh, fem, common, default_scalar_type
from dolfinx.common import Timer
import ufl
import basix

from dolfinx import io

from solvers import CustomLinearProblem, CustomNewtonSolver

import jax

jax.config.update("jax_enable_x64", True)


class ConvergenceError(Exception):
    pass

@dataclass
class SimulationParameters:
    thickness: float
    width: float
    height: float
    deg_u: int
    deg_quad: int
    indenter_radius: float
    velocity: float
    prescribed_displacement: float
    time_steps: int
    should_output: bool
    output_frequency: int = 1



def _create_quadrature_function_spaces(domain, deg_quad, vdim):
    W0e = basix.ufl.quadrature_element(
        domain.basix_cell(),
        value_shape=(),
        scheme="default",
        degree=deg_quad,
    )

    We = basix.ufl.quadrature_element(
        domain.basix_cell(),
        value_shape=(vdim,),
        scheme="default",
        degree=deg_quad,
    )

    WTe = basix.ufl.quadrature_element(
        domain.basix_cell(),
        value_shape=(vdim, vdim),
        scheme="default",
        degree=deg_quad,
    )
    WTe4 = basix.ufl.quadrature_element(
        domain.basix_cell(),
        value_shape=(vdim, vdim, vdim, vdim),
        scheme="default",
        degree=deg_quad,
    )

    W0 = fem.functionspace(domain, W0e)
    W = fem.functionspace(domain, We)
    WT = fem.functionspace(domain, WTe)
    WT4 = fem.functionspace(domain, WTe4)

    return W0,W,WT,WT4

def run_simulation(parameters: SimulationParameters, domain: mesh.Mesh, material: IsotropicMaterial | OrthotropicMaterial)-> tuple[list[float],list[float]]:
    # Create Function Space
    V = fem.functionspace(domain, ("Lagrange", parameters.deg_u,(domain.geometry.dim,)))
    
    # Boundary Conditions
    # Create center displacement boundary condition + clamped sides
    clamped_bc = boundary_conditions.clamped_boundary_condition(V,domain)
    center_point = np.array([parameters.width/2, parameters.height/2, parameters.thickness])
    def locate_center(x):
        return np.isclose(x[0], center_point[0], atol=1e-6) & \
               np.isclose(x[1], center_point[1], atol=1e-6) & \
               np.isclose(x[2], center_point[2], atol=1e-6)
    center_vertex = mesh.locate_entities(domain, 0, locate_center)
    center_dof_z = fem.locate_dofs_topological(V.sub(2), 0, center_vertex)
    center_bc, u_D_center = boundary_conditions.create_circular_boundary_condition(
        V, domain, thickness=parameters.thickness, center_x=center_point[0],
        center_y=center_point[1], radius=parameters.indenter_radius, displacement=0
    )
    u_D_center.value = default_scalar_type(0)
    bcs = [clamped_bc, center_bc]
    
    # Create Quadrature Function Spaces
    W0,W,WT,WT4 = _create_quadrature_function_spaces(domain, parameters.deg_quad, vdim = 3)

    # Create State Variables
    first_piola = fem.Function(WT, name="Stress")
    cauchy_stress = fem.Function(WT, name="Stress")
    piola_tangent = fem.Function(WT4, name="Tangent_operator")

    u = fem.Function(V, name="Total_displacement")
    du = fem.Function(V, name="Iteration_correction")

    u_old = fem.Function(V)
    v_old = fem.Function(V)
    a_old = fem.Function(V)
    a_new = fem.Function(V)
    velocity_new = fem.Function(V)

    # Define Time Descritization Parameters
    beta = fem.Constant(domain, 0.6)
    gamma = fem.Constant(domain, 0.5)
    dt = fem.Constant(domain, 0.0)

    a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
    a_expr = fem.Expression(a, V.element.interpolation_points)

    velocity = v_old + dt * ((1 - gamma) * a_old + gamma * a)
    velocity_expr = fem.Expression(velocity, V.element.interpolation_points)
    
    # Post Processing Variables
    von_mises_quadrature = fem.Function(W0)
    von_mises_projected_space = fem.functionspace(domain, ("CG", parameters.deg_u))
    von_mises_projected = fem.Function(von_mises_projected_space, name="von_Mises")

    # Variational Forms
    v = ufl.TestFunction(V)
    u_ = ufl.TrialFunction(V)

    def mass(u, v):
        return material.rho * ufl.dot(u, v) * ufl.dx

    dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": parameters.deg_quad, "quadrature_scheme": "default"},
    )
    Residual = mass(a,v) + ufl.inner(ufl.grad(v), first_piola) * dx
    i,j,k,l = ufl.indices(4)
    tangent_form = ufl.derivative(mass(a, v), u, u_) + ufl.inner(ufl.grad(v), ufl.as_tensor(
        piola_tangent[i,j,k,l] * ufl.grad(u_)[k,l],
        (i,j)
    )) * dx

    # Quadrature Point Handling
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, parameters.deg_quad)

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    ngauss = num_cells * len(weights)

    deformation_gradient_expr = fem.Expression(deformation_gradient(u), quadrature_points)

    def eval_at_quadrature_points(expression):
        return expression.eval(domain, cells).reshape(ngauss, -1)
    if material.material_type == MaterialType.Isotropic:
        tangent_operator_and_state = jax.jacfwd(
        first_piola_kirchhoff, argnums=0, has_aux=True
        )
        batched_constitutive_update = jax.jit(
        jax.vmap(tangent_operator_and_state, in_axes=(0, None, None))
        )
    elif material.material_type == MaterialType.Orthotropic:
        tangent_operator_and_state = jax.jacfwd(
        first_piola_kirchhoff_orthotropic(True), argnums=0, has_aux=True
        )
        batched_constitutive_update = jax.jit(
        jax.vmap(tangent_operator_and_state, in_axes=(0, None))
        )
    else:
        raise AssertionError(f"Wrong Material Type: {material.material_type}")
    
    # Constitutive Update
    def constitutive_update(u_current):
        with Timer("Constitutive update"):
            F_values = eval_at_quadrature_points(deformation_gradient_expr).reshape(ngauss, 3, 3)
            # Call appropriate kernel based on material type
            if material.material_type == MaterialType.Isotropic:
                Ct_values, state = batched_constitutive_update(F_values, material.mu, material.lamb)
            else:  # Orthotropic
                Ct_values, state = batched_constitutive_update(F_values, material.elasticity_matrix)
        
            piola_values, cauchy_values = state

            first_piola.x.array[:] = np.asarray(piola_values).ravel()
            cauchy_stress.x.array[:] = np.asarray(cauchy_values).ravel()
            piola_tangent.x.array[:] = np.asarray(Ct_values).ravel()
    
    # Tagent Problem + Newton Solver
    tangent_problem = CustomLinearProblem(
        tangent_form,
        -Residual,
        u=du,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="nonlin_plate",
    )
    newton = CustomNewtonSolver(tangent_problem)
    newton.callback = constitutive_update
    
    # Initialize State
    cauchy_stress.x.array[:] = 0.0
    first_piola.x.array[:] = 0.0
    piola_tangent.x.array[:] = 0.0
    u.x.array[:] = 0.0
    du.x.array[:] = 0.0

    u.x.scatter_forward()
    du.x.scatter_forward()

    v_old.x.array[:] = 0
    a_old.x.array[:] = 0

    final_time = parameters.prescribed_displacement / parameters.velocity 
    times_steps = np.linspace(0, final_time, parameters.time_steps)
    forces_center = [0]
    displacement_center = [0]
    v_old.x.array[center_dof_z] = -parameters.velocity
    simulation_time = 0.0

    if parameters.should_output:
        vtx_u = io.VTXWriter(domain.comm, "output/displacement.bp", [u], engine="BP4")
        vtx_vm = io.VTXWriter(domain.comm, "output/von_mises.bp", [von_mises_projected], engine="BP4")

    for i, delta_time in enumerate(np.diff(times_steps)):
        u_D_center.value -= parameters.velocity * delta_time
        dt.value = delta_time
        displacement_center.append(abs(u_D_center.value))
        simulation_time +=delta_time

        with Timer("Nonlinear solve"):
            niter, converged = newton.solve(u, tol=1e-6, Nitermax=50, verbose=parameters.should_output)

        if not converged:
            raise ConvergenceError(f"Solution did not converge: {converged}\\Parameters: {parameters}\nMaterial: {material}")
        
        # Store Results
        reaction = newton.get_reaction_center_bc(center_bc)
        forces_center.append(-reaction) 
        
        # Update old variables
        # Calculate Velocities and Accelerations
        velocity_new.interpolate(velocity_expr)
        a_new.interpolate(a_expr)
        v_old.x.array[:] = velocity_new.x.array[:]
        a_old.x.array[:] = a_new.x.array[:]
        u_old.x.array[:] = u.x.array[:] # Should be last

        # Output
        if parameters.should_output:
            print("Boundary_value:", u_D_center.value)
            print(f"Converged: {converged}")
            print(f"Z reaction = {reaction}")
            print("Displacement: ", u.x.array[center_dof_z])
            print("Velocity: ", v_old.x.array[center_dof_z])
            print("Acceleration: ", a_old.x.array[center_dof_z])
            postprocess_von_mises(domain,cauchy_stress,von_mises_quadrature,von_mises_projected,
                                  parameters.deg_u, parameters.deg_quad, u, plot=False)
            if i % parameters.output_frequency == 0 or i == len(times_steps) - 1:
                vtx_u.write(simulation_time)
                vtx_vm.write(simulation_time)

    # Post Processing
    if parameters.should_output:
        common.list_timings(MPI.COMM_WORLD)
        vtx_u.close()
        vtx_vm.close()
        # Post Processing
        postprocess_von_mises(domain,cauchy_stress,von_mises_quadrature,von_mises_projected,
                              parameters.deg_u, parameters.deg_quad,u, plot=True)
        plot(V,u)
        plot_force_displacement(forces_center,displacement_center)
    return displacement_center, forces_center

def main():
    parameters = SimulationParameters(thickness=0.005999999999999999,
                                       width=0.4,
                                       height=0.4,
                                       deg_u=2,
                                       deg_quad=4,
                                       indenter_radius=0.009/2,
                                       velocity = 350,
                                       prescribed_displacement = 0.02,
                                       time_steps = 100,
                                       should_output=True)
    domain = create_mesh.create_mesh(mesh_size=0.03,thickness = parameters.thickness, height=parameters.height, width=parameters.width)
    visualize_mesh(domain)
    # material = OrthotropicMaterial(
    #     e11=62e9,
    #     e22=62e9,
    #     e33=2e9,
    #     nu12=0.35,
    #     nu13=0.30,
    #     nu23=0.30,
    #     g12=0.7e9,
    #     g13=0.6e9,
    #     g23=0.6e9,
    #     rho=1400
    # )
    material = IsotropicMaterial(
    young_modulus=210e9,
    rho=7860,
    poisson_ratio=0.33
    )
    print(run_simulation(parameters=parameters,domain=domain,material=material))

if __name__ == '__main__':
    main()