
import os
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
import numpy as np




# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

N_point = 8

radius = 1
z_circle = 0.

angle = 0
angle_jump = 2*np.pi/N_point

with open('out.cpt', 'w') as f:
    for i in range(N_point+1): 
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        z = z_circle
        angle += angle_jump

        print(x,",",y,",",z, file=f)  

# Create a B-Spline curve instance
curve = BSpline.Curve()

# Set up the curve
curve.degree = 2
curve.ctrlpts = exchange.import_txt("circle_points.cpt")

# generate knot vector
# curve.knotvector =  utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
curve.knotvector = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]# Set evaluation delta
curve.delta = 0.01

# Evaluate curve
curve.evaluate()

print(curve.derivatives(0.35))
# Plot the control point polygon and the evaluated curve
vis_comp = VisMPL.VisCurve2D()
curve.vis = vis_comp
curve.render()