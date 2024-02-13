from geomdl import BSpline as BS # New class name confliction!
from geomdl import utilities
from geomdl import operations
from geomdl import helpers
from geomdl.visualization import VisMPL 
import numpy as np
import matplotlib.pyplot as plt


class BSpline:

    def __init__(self, P, p=3, num_of_pts=100):

        """ This class builds the BSpline for given control points

        Args:
            P (list): x and y values of control points as a list
            p (int, optional): degree of the BSpline. Defaults to 3.
            num_of_pts (int, optional): number of points that will be generated on the curve. Defaults to 73.
        """

        curve = BS.Curve()
        curve.degree = p
        curve.ctrlpts = P
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.sample_size = num_of_pts
        curve.evaluate()

        self.P = P
        self.curve = curve
        self.degree = p
        self.U = curve.knotvector
        self.pts = curve.evalpts # list
        self.pts_array = np.array(curve.evalpts)
        self.num_of_pts = num_of_pts

        self.first_derivative_curve = operations.derivative_curve(curve)
        self.second_derivative_curve = operations.derivative_curve(self.first_derivative_curve)

    def get_curvature(self):

        """calculates curvature field of the BSpline

        Returns:
            np.array: curvature field
        """

        d = self.first_derivative_curve
        d_x = np.array(d.evalpts)[:, 0]
        d_y = np.array(d.evalpts)[:, 1]

        dd = self.second_derivative_curve
        dd_x = np.array(dd.evalpts)[:, 0]
        dd_y = np.array(dd.evalpts)[:, 1]

        k = (d_x*dd_y - dd_x*d_y)/(d_x**2 + d_y**2)**(3/2)

        return k
    
    def get_curvature_from_point(self, pt):
        """
        evaluates the curvature for given point pt(x0,y0)

        Parameters
        ----------
        pt : list
            Given point.

        Returns
        -------
        k : [-]
            Curvature of the given point

        """
        
        t = self.get_t_from_point(pt)
        
        curve = self.curve
        
        # Computation of all derivatives up to order 2
        derivatives = curve.derivatives(u=t, order = 2)
        
        d_x = derivatives[1][0] 
        d_y = derivatives[1][1]

        dd_x = derivatives[2][0]
        dd_y = derivatives[2][1]

        k = (d_x*dd_y - dd_x*d_y)/(d_x**2 + d_y**2)**(3/2)
        
        return k

    def get_displacement(self, j):

        """ calculates displacement field for given span j

        Args:
            j (int): knot span

        Returns:
            list: displacement field
        """
        u = np.linspace(min(self.U),max(self.U),self.num_of_pts)
        V = [helpers.basis_function_one(self.degree, self.U, j, i) for i in u]

        return V

    def get_displacement_from_point(self, j, pt):

        """ calculates displacement value of the given point in span j

        Args:
            j (int): knot span
            pt (list): corresponding point on the curve

        Returns:
            [float]: [displacement value]
        """

        knot = self.get_t_from_point(pt)
        V = helpers.basis_function_one(self.degree, self.U, j, knot)
        return V

    def get_t_from_point(self, pt):

        """Calculates knot value for given point

        Args:
            pt (list): x and y values of point as a list

        Returns:
            float: knot value of the point
        """

        pts = self.pts_array
        n_pts = self.num_of_pts

        u_m = np.max(self.U)
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

    def render(self):

        """Plots the BSpline curve with built in function in geomdl
        """

        vis_comp1 = VisMPL.VisCurve2D()
        self.curve.vis = vis_comp1
        self.curve.render()
    
    def plot_curve(self):

        """Plots the BSpline curve with its control points via matplotlib
        """

        plt.figure()
        P = np.array(self.P)
        plt.plot(P[:, 0], P[:, 1], 's')
        plt.plot(self.pts_array[:, 0], self.pts_array[:, 1], '-')
        plt.show()

if __name__ == '__main__':

    P = [[0.  , 0.  ],
        [ .2 , 0.  ],
        [ .4 , 0.  ],
        [ .4 ,  .05],
        [ .4 ,  .1 ],
        [ .5 ,  .1 ],
        [ .6 ,  .1 ],
        [ .6 ,  .05],
        [ .6 , 0.  ],
        [ .8 , 0.  ],
        [1.  , 0.  ]]

    newcurve = BSpline(P,p=3,num_of_pts=100)
    newcurve2 = BSpline(P,p=3,num_of_pts=50)
    x = newcurve.pts_array[:, 0]
    y = newcurve.pts_array[:, 1]
    t = newcurve.get_t_from_point(newcurve.pts[45])
    curvature = newcurve.get_curvature()
    print(curvature)
    print(newcurve.get_displacement_from_point(4, newcurve.pts[45]))
    k = newcurve.get_curvature_from_point(newcurve.pts[45])
    V = newcurve.get_displacement(4)
    V2 = newcurve2.get_displacement(4)
    
    plt.plot(np.linspace(0,1,len(curvature)), curvature)
    plt.show()
    
    # plt.plot(np.linspace(0,1,len(V2)), V2)
    
    # # plt.plot(x,y)
    # plt.show()
    # # newcurve.plot_curve()

    P2 = [[0.  , 0. , 0. ],
        [ .2 , 0. , 0. ],
        [ .4 , 0. , 0. ],
        [ .4 ,  .05, 0.],
        [ .4 ,  .1 , 0.],
        [ .5 ,  .1 , 0.],
        [ .6 ,  .1 , 0.],
        [ .6 ,  .05, 0.],
        [ .6 , 0.  , 0.],
        [ .8 , 0.  , 0.],
        [1.  , 0.  , 0.]]

    newcurve3 = BSpline(P2,p=3,num_of_pts=100)
    print(newcurve3.get_t_from_point(newcurve3.pts[45]))
    print(newcurve3.U)
    newcurve3.render()

