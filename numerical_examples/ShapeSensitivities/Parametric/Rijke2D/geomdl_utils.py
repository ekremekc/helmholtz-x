from geomdl import BSpline, utilities, helpers, operations
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace, set_bc, locate_dofs_topological, DirichletBC
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc

class ParametricCurveCylinderical:

    def __init__(self, Geometry, BoundaryTag, ControlPoints, degree=3, delta=0.001) :

        self.ControlPoints = ControlPoints
        self.degree = degree
        self.delta = delta
        self.BoundaryTag = BoundaryTag

        self.mesh = Geometry.mesh
        self.facet_tags = Geometry.facet_tags

        self._curve = None
        self._U = None
        

        self._V = None
        self._K = None

        self._MakeCurve()
        self.first_derivative_curve = operations.derivative_curve(self.curve)
        self.second_derivative_curve = operations.derivative_curve(self.first_derivative_curve)

    @property
    def curve(self):
        return self._curve

    @property
    def U(self):
        return self._U

    @property
    def V(self):
        self._DisplacementField()
        return self._V
    
    @property
    def K(self):
        self._Curvature()
        return self._K

    def _MakeCurve(self):
        # Create a B-Spline curve instance
        curve = BSpline.Curve()

        # Set up curve
        curve.degree = self.degree
        curve.ctrlpts = self.ControlPoints

        # Auto-generate knot vector
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = self.delta

        # Evaluate curve
        curve.evaluate()

        self._curve = curve
        self._U = curve.knotvector

    def _DisplacementField(self):
        
        mesh = self.mesh
        facet_tags = self.facet_tags
        points = self.ControlPoints
        curve = self.curve
        DisplacementField = [None] * len(points)


        for control_point_index in range(len(points)):

            u = np.linspace(min(curve.knotvector),max(curve.knotvector),len(curve.evalpts))
            V = [helpers.basis_function_one(curve.degree, curve.knotvector, control_point_index, i) for i in u]

            Q = VectorFunctionSpace(mesh, ("CG", 1))

            gdim = mesh.topology.dim

            def V_function(x):
                scaler = points[-1][2] # THIS MIGHT NEEDS TO BE FIXED.. Cylinder's control points should be starting from 0 to L on z-axis.
                V_poly = np.poly1d(np.polyfit(u*scaler, np.array(V), 10))
                theta = np.arctan2(x[1],x[0])  
                values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
                values[0] = V_poly(x[2])*np.cos(theta)
                values[1] = V_poly(x[2])*np.sin(theta)
                return values

            temp = Function(Q)
            temp.interpolate(V_function)
            temp.name = 'V'

            facets = facet_tags.indices[facet_tags.values == self.BoundaryTag]
            dbc = DirichletBC(temp, locate_dofs_topological(Q, gdim-1, facets))

            DisplacementField[control_point_index] = Function(Q)
            DisplacementField[control_point_index].vector.set(0)
            DisplacementField[control_point_index].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            set_bc(DisplacementField[control_point_index].vector,[dbc])

        self._V = DisplacementField
    
    def _Curvature(self):

        """calculates curvature field of the BSpline

        Returns:
            np.array: curvature field
        """

        d = self.first_derivative_curve
        d_x = np.array(d.evalpts)[:, 0]
        d_y = np.array(d.evalpts)[:, 1]
        d_z = np.array(d.evalpts)[:, 2]

        dd = self.second_derivative_curve
        dd_x = np.array(dd.evalpts)[:, 0]
        dd_y = np.array(dd.evalpts)[:, 1]
        dd_z = np.array(dd.evalpts)[:, 2]
        
        # 3D curvature formula
        #https://en.wikipedia.org/wiki/Curvature#:~:text=for%20a%20parametrically-defined%20space%20curve%20in%20three%20dimensions%20given%20in%20cartesian%20coordinates%20by
        numerator = np.sqrt((dd_z*d_y - dd_y*d_z)**2 +  (dd_x*d_z - dd_z*d_x)**2 + (dd_y*d_x - dd_x*d_y)**2)
        denominator = (d_x**2 + d_y**2 + d_z**2)**(3/2)
        k = numerator/denominator
        print(k)
        # plt.plot(k)
        # plt.show()
        
        #
        u = np.linspace(min(self.U),max(self.U),len(self.curve.evalpts))
        mesh= self.mesh
        facet_tags = self.facet_tags

        V = FunctionSpace(mesh, ("CG", 1))

        gdim = mesh.topology.dim

        def K_function(x):
            scaler = self.ControlPoints[-1][2] # THIS MIGHT NEEDS TO BE FIXED.. Cylinder's control points should be starting from 0 to L on z-axis.
            K_poly = np.poly1d(np.polyfit(u*scaler, k, 10))
             
            values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
            print(gdim, x.shape[0])
            values = K_poly(x[2])
            return values

        temp = Function(V)
        temp.interpolate(K_function)
        temp.name = 'K'

        facets = facet_tags.indices[facet_tags.values == self.BoundaryTag]
        dbc = DirichletBC(temp, locate_dofs_topological(V, gdim-1, facets))

        Curvature = Function(V)
        Curvature.vector.set(0)
        Curvature.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        

        set_bc(Curvature.vector,[dbc])

        self._K = Curvature


    def get_t_from_point(self, pt):

        """Calculates knot value for given point

        Args:
            pt (list): x and y values of point as a list

        Returns:
            float: knot value of the point
        """
        
        pts = np.array(self.curve.evalpts)
        n_pts = len(self.curve.evalpts)

        u_m = np.max(self.curve.knotvector)
        min_dist = 1e3

        for i in range(n_pts):
            dist = np.linalg.norm(pt - pts[i])

            if dist < min_dist:
                min_dist = dist
                j = i

                if dist < 1e-14:  ##
                    t = u_m * i/(n_pts - 1)
                    return t

        if j == 0:            
            overall = np.linalg.norm(pts[1] - pts[0])
            frac = min_dist/overall
            t = u_m * frac/(n_pts - 1)

        elif j == n_pts - 1:
            overall = np.linalg.norm(pts[-1] - pts[-2])
            frac = min_dist/overall
            t = u_m * (j - frac)/(n_pts - 1)

        else:
            back = np.linalg.norm(pt - pts[j-1])
            forth = np.linalg.norm(pts[j+1] - pt)

            if back > forth:
                # pt is ahead of pts[j]
                overall = np.linalg.norm(pts[j+1] - pts[j])
                frac = min_dist/overall
                t = u_m * (j + frac)/(n_pts - 1)

            else:
                # pt is behind pts[j]
                overall = np.linalg.norm(pts[j] - pts[j-1])
                frac = min_dist/overall
                t = u_m * (j - frac)/(n_pts - 1)

        return t

if __name__ == '__main__':
    
    pass