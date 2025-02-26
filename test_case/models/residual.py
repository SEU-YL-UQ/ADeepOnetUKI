import jax.numpy as jnp 
from jax import grad


def residual_diffusion(self, params, u, *sensors):
    s = self.operator_net(params, u, *sensors)
    s_t = grad(self.operator_net, argnums=4)(params, u, *sensors)
    s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, *sensors)
    s_yy= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, *sensors)
    res = s_t - 0.01 * (s_xx + s_yy) - 0.01 * s**2 
    return res

def residual_heat(self, params, u, *sensors):
    s = self.operator_net(params, u, *sensors)
    s_t = grad(self.operator_net, argnums=4)(params, u, *sensors)
    s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, *sensors)
    s_yy= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, *sensors)
    res = s_t - s_xx - s_yy 
    return res 

def residual_flow(self, params, u, *sensors):
    s_x= grad(self.operator_net, argnums=2)(params, u, sensors[0], sensors[1])
    s_y= grad(self.operator_net, argnums=3)(params, u, sensors[0], sensors[1])
    s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, sensors[0], sensors[1])
    s_yy= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, sensors[0], sensors[1])
    res = -jnp.exp(sensors[2])*(s_xx + s_yy + s_x*sensors[3] + s_y*sensors[4])
    return res 


        