import numpy as np
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc
import dolfinx.la.petsc
from dolfinx.fem.petsc import assemble_vector

class CustomLinearProblem(fem.petsc.LinearProblem):
    def assemble_rhs(self, u=None):
        """
        Assemble right-hand side and apply lifting / Dirichlet conditions.

        Parameters
        ----------
        u : dolfinx.fem.Function, optional
            Current Newton iterate. When provided, the RHS is assembled for the
            correction problem using the difference between prescribed Dirichlet
            values and the current iterate on constrained dofs.
        """
        with self._b.localForm() as b_loc:
            b_loc.set(0.0)

        fem.petsc.assemble_vector(self._b, self._L)

        x0 = [] if u is None else [u.x.petsc_vec]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, alpha=1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        x0 = None if u is None else u.x.petsc_vec
        fem.petsc.set_bc(self._b, self.bcs, x0, alpha=1.0)

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        """
        Solve linear system for the Newton correction and copy PETSc solution
        back into the dolfinx Function.
        """
        self._solver.solve(self._b, self._x)

        # PETSc vector -> ghost update
        dolfinx.la.petsc._ghost_update(
            self._x,
            PETSc.InsertMode.INSERT,
            PETSc.ScatterMode.FORWARD,
        )

        # PETSc vector -> fem.Function
        dolfinx.fem.petsc.assign(self._x, self.u)


class CustomNewtonSolver:
    def __init__(self, tangent_problem):
        self.tangent_problem = tangent_problem
        self.du = self.tangent_problem.u
        self.max_step_reductions = 15
        
    def callback(self, u, *args):
        pass

    def solve(self, u, *args, tol=1e-8, Nitermax=50, verbose=True):
        converged = False

        for k in range(Nitermax):
            self.callback(u, *args)

            self.tangent_problem.assemble_lhs()
            self.tangent_problem.assemble_rhs(u)

            res_norm = self.tangent_problem._b.norm()
            if k == 0:
                if res_norm > 1e-6:
                    res0 = res_norm
                else:
                    res0 = 1

            rel_norm = res_norm / res0

            if verbose:
                du_norm = np.linalg.norm(self.du.x.array)
                print(
                    f"Newton iter {k:2d} |R| = {res_norm:.6e} "
                    f"|R|/|R0| = {rel_norm:.6e} "
                    f"|du| = {du_norm:.6e} "
                    f"du/u = {du_norm / (u.x.petsc_vec.norm()+1e-12):.6e}"
                )

            if rel_norm < tol:
                converged = True
                break
            # print("---- BEFORE SOLVE ----")
            # print("u min/max:", u.x.array.min(), u.x.array.max())
            # print("du min/max:", self.du.x.array.min(), self.du.x.array.max())
            # print("du finite:", np.isfinite(self.du.x.array).all())
            # A = self.tangent_problem._A
            # print("A norm:", A.norm())
            self.tangent_problem.solve_system()
            # self.limit_displacement()
            #self.line_search(u, self.du.x.array, res_norm)
            u.x.array[:] += self.du.x.array[:]* min(0.8,20*(k+1)/Nitermax)
            u.x.scatter_forward()
            
            # print("---- AFTER SOLVE ----")
            # print("du min/max:", self.du.x.array.min(), self.du.x.array.max())
            # print("du finite:", np.isfinite(self.du.x.array).all())
        return k + 1, converged
    
    def limit_displacement(self, max_displacement=0.05):
        """Limit displacement increment to maximum value"""
        du_norm = np.linalg.norm(self.du.x.array)
        if du_norm > max_displacement:
            scale = max_displacement / du_norm
            self.du.x.array[:] *= scale

    def compute_residual_norm(self, u):
        """Compute residual norm for given u"""
        # Temporarily set u
        u_current = self.tangent_problem.u.x.array.copy()
        self.tangent_problem.u.x.array[:] = u
        self.tangent_problem.u.x.scatter_forward()
        
        # Assemble residual
        self.tangent_problem.assemble_rhs(self.tangent_problem.u)
        res_norm = self.tangent_problem._b.norm()
        
        # Restore
        self.tangent_problem.u.x.array[:] = u_current
        self.tangent_problem.u.x.scatter_forward()
        
        return res_norm
    
    def line_search(self, u, du, current_residual):
        """Find step length that reduces residual"""
        alpha = 1.0
        original_u = u.x.array.copy()
        
        for i in range(self.max_step_reductions):
            # Try this step
            u.x.array[:] = original_u + alpha * du
            u.x.scatter_forward()
            
            # Compute new residual
            self.tangent_problem.assemble_rhs(u)
            new_residual = self.tangent_problem._b.norm()
            
            # Check if residual decreased
            if new_residual < current_residual:
                return alpha, new_residual
            
            # Cut step in half
            alpha *= 0.5
        
        # If we get here, take smallest step
        u.x.array[:] = original_u + alpha * du
        u.x.scatter_forward()
        return alpha, new_residual
        
    def get_reactions(self, bcs):
        """Get reaction forces after solve using the official DOLFINx API."""

        # 1. Assemble the linear form (-Residual)
        L_form = self.tangent_problem._L
        R_vec = -assemble_vector(fem.form(L_form))
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # 2. Get function space info
        V = self.tangent_problem.u.function_space
        dim = V.mesh.geometry.dim
        component_names = ['x', 'y', 'z'][:dim]

        reactions = {}
        for i, bc in enumerate(bcs):
            dofs = bc.dof_indices()[0]  # Returns (dof_array, num_owned)
            
            if len(dofs) == 0:
                continue
            
            reaction_vals = R_vec.array[dofs]
            
            # Sum by component (x, y, z)
            forces = {}
            for comp, name in enumerate(component_names):
                comp_dofs = dofs[comp::dim]
                if len(comp_dofs) > 0:
                    comp_vals = reaction_vals[comp::dim]
                    forces[f'F{name}'] = float(np.sum(comp_vals))
                else:
                    forces[f'F{name}'] = 0.0
            
            forces['total'] = float(np.sqrt(sum(f**2 for f in forces.values())))
            reactions[f"BC_{i}"] = forces
        
        return reactions
    def get_all_nodal_reactions(V, tangent_problem):
        """
        Get reactions at ALL nodes (useful for visualization)
        """
        
        # Assemble residual
        L_form = tangent_problem._L
        R_vec = -assemble_vector(fem.form(L_form))
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Get coordinates of all nodes
        coords = V.tabulate_dof_coordinates()
        dim = V.mesh.geometry.dim
        
        # Group reactions by node
        node_reactions = {}
        
        # Each DOF belongs to a node
        # In Lagrange space, each node has 'dim' DOFs
        num_nodes = len(coords)
        
        for node in range(num_nodes):
            dof_start = node * dim
            dof_end = dof_start + dim
            
            if dof_end <= len(R_vec.array):
                reaction_vals = R_vec.array[dof_start:dof_end]
                
                # Only store if reaction is significant
                if np.any(np.abs(reaction_vals) > 1e-12):
                    node_reactions[tuple(coords[node])] = {
                        'Fx': float(reaction_vals[0]),
                        'Fy': float(reaction_vals[1]) if dim > 1 else 0,
                        'Fz': float(reaction_vals[2]) if dim > 2 else 0,
                        'total': float(np.linalg.norm(reaction_vals))
                    }
        
        return node_reactions
    def get_all_nodal_reactions(self, V):
        """
        Get reactions at ALL nodes (useful for visualization)
        """
        
        # Assemble residual
        L_form = self.tangent_problem._L
        R_vec = assemble_vector(fem.form(L_form))
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Get coordinates of all nodes
        coords = V.tabulate_dof_coordinates()
        dim = V.mesh.geometry.dim
        
        # Group reactions by node
        node_reactions = {}
        
        # Each DOF belongs to a node
        # In Lagrange space, each node has 'dim' DOFs
        num_nodes = len(coords)
        
        for node in range(num_nodes):
            dof_start = node * dim
            dof_end = dof_start + dim
            
            if dof_end <= len(R_vec.array):
                reaction_vals = R_vec.array[dof_start:dof_end]
                
                # Only store if reaction is significant
                if np.any(np.abs(reaction_vals) > 1e-12):
                    node_reactions[tuple(coords[node])] = {
                        'Fx': float(reaction_vals[0]),
                        'Fy': float(reaction_vals[1]) if dim > 1 else 0,
                        'Fz': float(reaction_vals[2]) if dim > 2 else 0,
                        'total': float(np.linalg.norm(reaction_vals))
                    }
        
        return node_reactions
    
    def get_reaction_center_bc(self,bc):
        """Get reaction forces after solve using the official DOLFINx API."""

        # 1. Assemble the linear form (-Residual)
        L_form = self.tangent_problem._L
        R_vec = -assemble_vector(fem.form(L_form))
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        dofs = bc.dof_indices()[0]  # Returns (dof_array, num_owned)

        reaction_vals = R_vec.array[dofs]

        return sum(reaction_vals)