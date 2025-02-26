from fenics import *
import numpy as np
import jax.numpy as jnp
from jax import random
import fenics_adjoint
set_log_active(False)



class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0

class Source(UserExpression):
    def __init__(self, m, t, T_l, s, sigma, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.m = m 
        self.t_l = T_l 
        self.s = s 
        self.sigma = sigma
    
    def eval(self, value, x):
        r = np.sqrt((x[0] - self.m[0])**2 + (x[1] - self.m[1])**2)
        H = 1.0 if self.t > self.t_l else 0
        value[0] = self.s / (2 * np.pi * self.sigma**2) * np.exp(-r**2 / (2 * self.sigma**2)) * (1 - H)
    
    def value_shape(self):
        return ()
            
        
             

class Solver:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def forward_flow(self, a, V, derivative = False, compute = True):
        fenics_adjoint.set_working_tape(fenics_adjoint.Tape())
        a = a.flatten()
        mesh = V.mesh()
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, Constant(0), boundary)
        m = Function(V)
        m.vector()[:] = a[dof_to_vertex_map(V)]

        u = TrialFunction(V)
        v = TestFunction(V)
        f_val = self.f(V)

        a = m*inner(grad(u), grad(v)) * dx
        L = f_val*v*dx
        u = Function(V)
        solve(a == L, u, bc)
    
        if derivative:
            return u, m
        elif compute:
            return u.compute_vertex_values(mesh)
        else:
            return u
    
    @classmethod
    def forward_possion(self, a, V, derivative = False, compute = True):
        fenics_adjoint.set_working_tape(fenics_adjoint.Tape())
        a = a.flatten()
        mesh = V.mesh()
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(1), boundary)
        f = Function(V)
        f.vector()[:] = a[dof_to_vertex_map(V)]

        u = TrialFunction(V)
        v = TestFunction(V)


        a = inner(grad(u), grad(v)) * dx
        L = f*v*dx
        u = Function(V)
        solve(a == L, u, bc)
    
        if derivative:
            return u, f
        elif compute:
            return u.compute_vertex_values(mesh)
        else:
            return u
    
    
    @staticmethod
    def f(V):
        m = Function(V)
        nodes = V.mesh().coordinates()
        f = np.zeros(V.dim())
        space_dim = V.mesh().geometric_dimension()
        if space_dim == 2:
            f[(nodes[:,1]>=0) & (nodes[:,1]<=2/3)] = 1000
            f[(nodes[:,1]>2/3) & (nodes[:,1]<=5/6)] = 2000
            f[(nodes[:,1]>5/6) & (nodes[:,1]<=1)] = 3000
        elif space_dim == 1:
            f[(nodes[:,0]>=0) & (nodes[:,0]<=1/2)] = 1000
            f[(nodes[:,0]>1/2) & (nodes[:,0]<=1)] = 2000
        m.vector()[:] = f[dof_to_vertex_map(V)]
        return m
    
    
        
    
    @classmethod
    def forward_heat(self, a, V, num_t = 30, derivative = False):
        fenics_adjoint.set_working_tape(fenics_adjoint.Tape())
        a = a.flatten()
        dt = 1 / num_t
        mesh = V.mesh()
        # Create mesh and define function space
        if mesh.geometric_dimension() == 1:
            initial_condition = Expression('100*sin(x[0])', degree = 1)
        else:
            initial_condition = Expression('100*sin(x[0])*sin(x[1])', degree = 2)
      
        def boundary(x, on_boundary):
            return on_boundary
    
        bc = DirichletBC(V, Constant(0), boundary)
        u_n = interpolate(initial_condition, V)
        
        changing_term = Expression("exp(-t)", t = 0, degree = 2)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Function(V)
        f.vector()[:] = a[dof_to_vertex_map(V)]

        F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f*changing_term)*v*dx
        a, L = lhs(F), rhs(F)

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(num_t):
            t += dt
            changing_term.t = t
            solve(a == L, u, bc)
            u_n.assign(u)
        u_T = u_n.compute_vertex_values(mesh)
        if derivative:
            return u_n, f
        return u_T
    
    @classmethod
    def forward_diffusion(self, a, V, num_t = 30, derivative = False):
        fenics_adjoint.set_working_tape(fenics_adjoint.Tape())
        a = a.flatten()
        dt = 1 / num_t
        mesh = V.mesh()
        u_n = Function(V)
        u_n.vector()[:] = a[dof_to_vertex_map(V)]
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        pressure = Expression(('sin(pi*x[0])*cos(pi*x[1])', '-cos(pi*x[0])*sin(pi*x[1])'), degree = 2)
        # pressure = computeVelocityField(mesh)
        a = 2*u*v*dx + dt/30*dot(grad(u),grad(v))*dx + dt*dot(pressure,grad(u))*v*dx
        L =  2*u_n*v*dx -dt/30*dot(grad(u_n),grad(v))*dx - dt*dot(pressure,grad(u_n))*v*dx

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(num_t):
            t += dt
            solve(a == L, u)
            u_n.assign(u)
        u_T = u_n.compute_vertex_values(mesh)
        return u_T
        
    @classmethod
    def forward_burgers(self, V, num_t = 30, derivative = False):
        """The solution is not right perhaps"""
        fenics_adjoint.set_working_tape(fenics_adjoint.Tape())
        # a = a.flatten()
        dt = 1/num_t 
        mesh = V.mesh()
        # u0 = Function(V)
        # u0.vector()[:] = a[dof_to_vertex_map(V)]
        u0 = interpolate(Expression('-sin(pi*x[0])', degree = 2), V)
        
        def left(x, on_boundary):
            return on_boundary and near(x[0], -1)
        def right(x, on_boundary):
            return on_boundary and near(x[0], 1)
        
        bc_l = DirichletBC(V, Constant(0), left)
        bc_r = DirichletBC(V, Constant(0), right)
        bcs = [bc_l, bc_r]
        
        u = Function(V)
        v = TestFunction(V)
        nu = Constant(0.01)
        F = ((u - u0)/dt)*v*dx + inner(dot(u, u.dx(0)), v)*dx + nu*inner(u.dx(0), v.dx(0))*dx 
        result = []
        result.append(u0.compute_vertex_values(mesh))
        
        t = 0
        for n in range(num_t):
            t += dt 
            solve(F == 0, u, bcs)
            u0.assign(u)
            result.append(u0.compute_vertex_values(mesh))
        
        return np.vstack(result)
    
    @classmethod
    def forward_source(self, m, V):
        dt = 0.005
        mesh = V.mesh()
        source_expr = Source(m, 0, 0.05, 5, 0.1, degree = 2)
        u_old = Function(V)
        v = TestFunction(V)
        u = TrialFunction(V)

        a = u*v*dx + dt*inner(grad(u), grad(v))*dx 
        L = u_old*v*dx + dt*source_expr*v*dx
        t = 0
        result = []
        u_sol = Function(V)
        while t <= 0.15:
            t += dt
            source_expr.t = t 
            L = u_old*v*dx + dt*source_expr*v*dx
            solve(a == L, u_sol)
            u_old.assign(u_sol)
            if np.isclose(t, 0.05) or np.isclose(t, 0.15):
                result.append(u_sol.compute_vertex_values(mesh))
        return result
        
        
        
    
    
     
   
    
        
    
    
    
        

        




    
        
            

    
    
