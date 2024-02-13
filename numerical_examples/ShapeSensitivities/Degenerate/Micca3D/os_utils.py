from datetime import datetime
import dolfin as dolf
import os
import pickle

from helmholtz_pkg.xdmf_utils import save_xdmf

from math import pi

# os.system('rm -r 2020*')


def create_dirs():

    my_dir = os.path.join(os.getcwd(), datetime.now().strftime('%Y%b%d-%H%M%S'))
    os.makedirs(my_dir)
    # print(my_dir)

    # ________________________________________________________________________________

    mesh_dir = os.path.join(my_dir, 'MeshDir')
    os.makedirs(mesh_dir)
    # print(mesh_dir)

    results_dir = os.path.join(my_dir, 'ResultsDir')
    os.makedirs(results_dir)
    # print(results_dir)

    # ________________________________________________________________________________

    eigenvectors_dir = os.path.join(results_dir, 'Eigenvectors')
    os.makedirs(eigenvectors_dir)
    # print(eigenvectors_dir)

    # shape_sensitivities_dir = os.path.join(results_dir, 'ShapeSensitivities')
    # os.makedirs(shape_sensitivities_dir)
    # print(shape_sensitivities_dir)

    pickle_dir = os.path.join(results_dir, 'Pickle')
    os.makedirs(pickle_dir)
    # print(pickle_dir)

    return my_dir, mesh_dir, results_dir, eigenvectors_dir, pickle_dir
    # return my_dir, mesh_dir, results_dir, eigenvectors_dir, shape_sensitivities_dir, pickle_dir


def create_iter_subdirs(mesh_dir, eigenvectors_dir, i):

    iter_subdir = "{0}{1:03d}".format('Iter', i)

    mesh_subdir = os.path.join(mesh_dir, iter_subdir)
    os.makedirs(mesh_subdir)

    eigenvectors_subdir = os.path.join(eigenvectors_dir, iter_subdir)
    os.makedirs(eigenvectors_subdir)

    return mesh_subdir, eigenvectors_subdir

# ________________________________________________________________________________


def save_eigenvector(subdir, base_name, data, i=None):
    """
    save data in subdir as base_name.ext or base_name_i.ext
    The extensions are xdmf (and h5), pvd (and vtu)
    :param subdir: subdirectory path
    :param base_name: base name of the file
    :param data: data to save
    :param i: iteration (optional)
    """

    ext = "xdmf"
    if i:
        filename = "{0}_{1:03d}.{1}".format(base_name, i, ext)
    else:
        filename = "{0}.{1}".format(base_name, ext)
    filename = os.path.join(subdir, filename)
    # print(filename)

    save_xdmf(filename, data)

    ext = "pvd"
    if i:
        filename = "{0}_{1:03d}.{2}".format(base_name, i, ext)
    else:
        filename = "{0}.{1}".format(base_name, ext)
    filename = os.path.join(subdir, filename)
    # print(filename)

    dolf.File(filename) << data


def pickle_dump(directory, filename, dictionary, i=None):
    if i or i == 0:
        filename = "{0}_{1:03d}.{2}".format(filename, i, "pickle")
    else:
        filename = "{0}.{1}".format(filename, "pickle")

    filename = os.path.join(directory, filename)

    with open(filename, "wb") as pickle_out:  # write bytes
        pickle.dump(dictionary, pickle_out)

#     with open(filename, "rb") as pickle_in:  # read bytes
#         dictionary = pickle.load(pickle_in)


def save_eigs_as_text(directory, dictionary, i=None, tol=1e-10):

    if i or i == 0:
        # filename = "{0}_{1:03d}.{2}".format("eigs", i, "txt")
        filename = "{0}_{1:03d}".format("eigs", i)
    else:
        # filename = "{0}.{1}".format("eigs", "txt")
        # filename = "eigs.txt"
        filename = "eigs"

    filename = os.path.join(directory, filename)

    # formatting
    s = '{:.0e}'.format(tol)
    # '1e-10'
    s = int(s[-2:])
    # 10
    s = '{{:+.{}f}}'.format(s)
    # '{:+.10f}'

    with open(filename, "w") as out:
        for key, value in dictionary.items():
            out.write("{}\n".format(key))
            out.write("eigenvalue = {}  {}j\n".format(s.format(value.real), s.format(value.imag)))
            value /= 2 * pi
            out.write("frequency  = {}  {}j\n\n".format(s.format(value.real), s.format(value.imag)))


def pickle_load(filename):
    with open(filename, 'rb') as pickle_in:
        d = pickle.load(pickle_in)
    return d



