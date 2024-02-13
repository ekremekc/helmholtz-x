import datetime
start_time = datetime.datetime.now()
import gmsh
import dolfinx
from dolfinx.fem import Function, FunctionSpace, form
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from ufl import Measure,  TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc
import dolfinx.io

model_rank = 0
mesh_comm = MPI.COMM_WORLD

gmsh.initialize()
if mesh_comm.rank == model_rank:
    
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(__name__)
    lc = 3e-2
    L = 1
    h = 0.5
    x_f = 0.5
    a_f = 0.5
    geom = gmsh.model.geo
    
    # Upstream Domain
    """ p4  _____ p3
           |    |
           |    |
        p1 |____| p2
    
    """
    
    p1 = geom.addPoint(0, -h, 0, lc)
    p2 = geom.addPoint((x_f ), -h, 0, lc)
    p3 = geom.addPoint((x_f ), h, 0, lc)
    p4 = geom.addPoint(0, h, 0, lc)

    l1 = geom.addLine(1, 2)
    l2 = geom.addLine(2, 3)
    l3 = geom.addLine(3, 4)
    l4 = geom.addLine(4, 1)

    ll1 = geom.addCurveLoop([1, 2, 3, 4])
    s1 = geom.addPlaneSurface([1])
    
    # Subdomain (Flame)
    """ p3  _____ p6
           |    |
           |    |
        p2 |____| p5
    
    """
    
    p5 = geom.addPoint((x_f + a_f), -h, 0, lc)
    p6 = geom.addPoint((x_f + a_f), +h, 0, lc)

 
    l5 = geom.addLine(2, 5)
    l6 = geom.addLine(5, 6)
    l7 = geom.addLine(6, 3)


    ll2 = geom.addCurveLoop([5, 6, 7, -2])
    s2 = geom.addPlaneSurface([2])

    gmsh.model.geo.synchronize()

    #Whole geometry
    gmsh.model.addPhysicalGroup(2, [1], 1)
    #Flame Tag
    gmsh.model.addPhysicalGroup(2, [2], 2)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()

from dolfinx.io.gmshio import model_to_mesh
mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
gmsh.finalize()

dx = Measure("dx",domain=mesh,subdomain_data=ct)

V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

sos = 347
a_form = form(sos**2*inner(grad(u), grad(v))*dx - sos**2*inner(grad(u), grad(v))*dx(2))
A = assemble_matrix(a_form)
A.assemble()

L = 0.02706
U = 8.27
b_form = form(sos**2 * 1j*L/U*inner(grad(u),grad(v))*dx(2))
B = assemble_matrix(b_form)
B.assemble()

c_form = form(-inner(u , v) * dx)
C = assemble_matrix(c_form)
C.assemble()

target = (180*2*np.pi)

solver = SLEPc.PEP().create(MPI.COMM_WORLD)
operators = [A, B, C]
solver.setOperators(operators)
# spectral transformation
st = solver.getST()
st.setType('sinvert')
solver.setTarget(target)
solver.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
solver.setTolerances(1e-15)
solver.setFromOptions()

solver.solve()

A = solver.getOperators()[0]
vr, vi = A.createVecs()

eig = solver.getEigenpair(0, vr, vi)
omega = eig

print(omega)

p = Function(V)
p.vector.setArray(vr.array)
p.x.scatter_forward()

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)