import fenics
import fenics_adjoint
import numpy as np

__all__ = ['fenics_to_numpy']

def fenics_to_numpy(fenics_var):
    """Convert FEniCS variable to numpy array"""
    if isinstance(fenics_var, (fenics.Constant, fenics_adjoint.Constant)):
        return fenics_var.values()

    if isinstance(fenics_var, (fenics.Function, fenics_adjoint.Constant)):
        np_array = fenics_var.vector().get_local()
        n_sub = fenics_var.function_space().num_sub_spaces()
        # Reshape if function is multi-component
        if n_sub != 0:
            np_array = np.reshape(np_array, (len(np_array) // n_sub, n_sub))
        return np_array

    if isinstance(fenics_var, fenics.GenericVector):
        return fenics_var.get_local()

    if isinstance(fenics_var, fenics_adjoint.AdjFloat):
        return np.array(float(fenics_var), dtype=np.float_)

    raise ValueError('Cannot convert ' + str(type(fenics_var)))