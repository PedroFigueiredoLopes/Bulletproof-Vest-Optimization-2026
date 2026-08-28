from dolfinx.plot import vtk_mesh
import pyvista
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
import basix



def visualize_mesh(domain):
    """Quick mesh visualization using PyVista"""
    # Get mesh geometry
    cells, cell_types, geometry = vtk_mesh(domain, domain.topology.dim)
    
    # Create PyVista grid
    grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
    
    # Add cell data for coloring (optional)
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    
    # Plot
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
    
    return grid

def postprocess_von_mises(domain, cauchy_stress, vm_quad, vm_cg, deg_u, deg_quad, u=None, plot=True):
    """
    Standard FEniCSx postprocessing for von Mises stress with warped mesh visualization.
    
    Args:
        domain: The mesh
        cauchy_stress: Function in quadrature space (WT) containing Cauchy stress (3x3 tensor)
        vm_quad: Scalar function in quadrature space (temporary)
        vm_cg: Scalar function in CG space (for visualization)
        deg_u: Degree of displacement space (for CG space)
        deg_quad: Quadrature degree
        u: Optional displacement for warped visualization (Function in appropriate space)
    """
    # Step 1: Project Cauchy stress to CG space (tensor)
    V_stress_cg = fem.functionspace(domain, ("CG", vm_cg.function_space.ufl_element().degree, (3, 3)))
    stress_cg = fem.Function(V_stress_cg)
    
    # Project using L2 projection with quadrature
    v_test = ufl.TestFunction(V_stress_cg)
    u_trial = ufl.TrialFunction(V_stress_cg)
    dx_q = ufl.dx(metadata={"quadrature_degree": deg_quad})
    
    a = ufl.inner(u_trial, v_test) * dx_q
    L = ufl.inner(cauchy_stress, v_test) * dx_q
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options_prefix="projection")
    stress_cg = problem.solve()
    
    # Step 2: Compute von Mises at CG points from projected stress tensor
    stress_cg_array = stress_cg.x.array.reshape(-1, 3, 3)
    trace = np.trace(stress_cg_array, axis1=1, axis2=2)
    s = stress_cg_array - trace[:, None, None] / 3 * np.eye(3)
    von_mises_cg = np.sqrt(1.5 * np.sum(s * s, axis=(1, 2)))
    vm_cg.x.array[:] = von_mises_cg
    
    # Step 3: Visualize
    if plot:
        # Create mesh for CG function space (continuous vertices)
        topology, cell_types, geometry = vtk_mesh(vm_cg.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # Add von Mises stress (point data)
        grid.point_data["von_Mises"] = vm_cg.x.array
        
        # Optional: Warp by displacement if provided
        if u is not None:
            # Get displacement in CG space (matching the mesh vertices)
            V_u = fem.functionspace(domain, ("CG", deg_u, (3,)))
            u_vec = fem.Function(V_u)
            u_vec.interpolate(u)
            
            # Add displacement vectors to grid points
            grid["u"] = u_vec.x.array.reshape((geometry.shape[0], 3))
            
            # Warp the grid by displacement
            grid_warped = grid.warp_by_vector("u", factor=1)
            
            # Plot warped mesh with von Mises stress
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid_warped, scalars="von_Mises", cmap="viridis", 
                           show_edges=True, smooth_shading=True)
            plotter.view_xy()
            plotter.show()
        else:
            # Plot unwarped mesh
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, scalars="von_Mises", cmap="viridis", 
                           show_edges=True, smooth_shading=True)
            plotter.show()
    
    return vm_cg

def von_mises_dg(domain, deg_quad,cauchy_stress, ngauss,deg_u,u, von_mises_cell, plot=True):
     # 1. Get quadrature points and reshape Cauchy stress
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)
    
    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    
    # 2. Reshape Cauchy stress to tensor form (ngauss, 3, 3)
    cauchy_tensor = cauchy_stress.x.array.reshape(ngauss, 3, 3)
    
    # 3. Compute von Mises at each quadrature point
    von_mises_qp = np.zeros(ngauss)
    for i in range(ngauss):
        sigma = cauchy_tensor[i]
        trace = np.trace(sigma)
        s = sigma - (trace / 3.0) * np.eye(3)
        von_mises_qp[i] = np.sqrt(1.5 * np.sum(s * s))
    
    von_mises_reshaped = von_mises_qp.reshape(num_cells, len(weights))
    cell_averages = np.average(von_mises_reshaped, axis=1, weights=weights)
    von_mises_cell.x.array[:len(cell_averages)] = cell_averages
    if plot:        
        # Get mesh topology for DG-0 (cell centers)
        cells_dg0, cell_types_dg0, geometry_dg0 = vtk_mesh(domain, 3)
        grid = pyvista.UnstructuredGrid(cells_dg0, cell_types_dg0, geometry_dg0)
        grid.cell_data["von_Mises"] = von_mises_cell.x.array
        V_u = fem.functionspace(domain, ("CG", deg_u, (3,)))
        u_vec = fem.Function(V_u)
        u_vec.interpolate(u)
    
        grid["u"] = u_vec.x.array.reshape((geometry_dg0.shape[0], 3))
        grid = grid.warp_by_vector("u", factor=1)
        # Add to plotter
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid,scalars="von_Mises", show_edges=True)
        plotter.view_xy()
        plotter.show()
    
def plot(V, uh):
    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.show()

    # Create plotter and pyvista grid
    p = pyvista.Plotter()
    topology, cell_types, geometry = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
    warped = grid.warp_by_vector("u", factor=1)
    actor_1 = p.add_mesh(warped,show_edges=True)
    p.show_axes()   
    p.show()

def plot_force_displacement(forces,displacement):
    fig, ax = plt.subplots()
    ax.plot(displacement[1:],forces[1:])
    plt.show()

def post_process_von_mises_dg(domain, cauchy_stress, vm_quad, vm_dg, deg_u, deg_quad, u=None, plot=True):
    # Get quadrature data
    ngauss = len(cauchy_stress.x.array) // 9
    cauchy_tensor = cauchy_stress.x.array.reshape(ngauss, 3, 3)
    
    # Step 1: Compute von Mises at quadrature points (vectorized)
    trace = np.trace(cauchy_tensor, axis1=1, axis2=2)
    s = cauchy_tensor - trace[:, None, None] / 3 * np.eye(3)
    von_mises_qp = np.sqrt(1.5 * np.sum(s * s, axis=(1, 2)))
    vm_quad.x.array[:] = von_mises_qp
    
    # L2 projection using same quadrature degree
    dx_q = ufl.dx(metadata={"quadrature_degree": deg_quad})
    v_test = ufl.TestFunction(vm_dg.function_space)
    u_trial = ufl.TrialFunction(vm_dg.function_space)
    print("Min quadrature value:",np.min(vm_quad.x.array))
    a = u_trial * v_test * dx_q
    L = vm_quad * v_test * dx_q
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[],     petsc_options={
        "ksp_type": "gmres",           # Krylov method (or "preonly" for direct)
        "pc_type": "lu",               # Direct solver (most accurate)
        "pc_factor_mat_solver_type": "mumps",  # Use MUMPS direct solver
        "ksp_monitor": None,           # Print residual at each iteration [citation:1]
        "ksp_converged_reason": None,  # Print why solver stopped
        "ksp_rtol": 1e-12,             # Very tight tolerance (important!)
        "ksp_atol": 1e-15,
        "ksp_max_it": 1000,
    },petsc_options_prefix="projection")
    vm_dg_new = problem.solve()
    vm_dg.x.array[:] = vm_dg_new.x.array[:]
    print(f"Min value: {np.min(vm_dg.x.array)}")
    print(f"Max value: {np.max(vm_dg.x.array)}")
    # Step 4: Visualize
    if plot:
        topology, cell_types, geometry = vtk_mesh(vm_dg.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["von_Mises"] = vm_dg.x.array
        
        # Optional: Warp by displacement if provided
        if u is not None:
            V_u = fem.functionspace(domain, ("CG", deg_u, (3,)))
            u_vec = fem.Function(V_u)
            u_vec.interpolate(u)
        
            grid["u"] = u_vec.x.array.reshape((geometry.shape[0], 3))
            grid = grid.warp_by_vector("u", factor=1)
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, cmap="viridis", show_edges=True, smooth_shading=False)
        plotter.show()
    return vm_dg