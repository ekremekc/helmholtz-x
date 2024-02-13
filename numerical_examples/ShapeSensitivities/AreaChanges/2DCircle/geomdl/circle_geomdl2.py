from geomdl.visualization import VisMPL
from geomdl.shapes import curve2d


# Generate a NURBS full circle from 9 control points
circle = curve2d.full_circle(radius=5.0)
circle.sample_size = 100

print(circle.ctrlpts)

print(circle.derivatives(0.35))
# Render the circle and the control points polygon
vis_config = VisMPL.VisConfig(ctrlpts=True, figure_size=[8, 8])
vis_comp = VisMPL.VisCurve2D(config=vis_config)
circle.vis = vis_comp
circle.render()