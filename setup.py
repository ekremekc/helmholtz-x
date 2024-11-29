from setuptools import setup

setup(
    name='helmholtz_x',
    version = '0.9.0',
    author='Ekrem Ekici',
    author_email='ee331@cantab.ac.uk',
    packages=['helmholtz_x'],
    install_requires=[
        'h5py==3.12.1',
        'meshio==5.3.5',
        'numba==0.60.0',
        'numpy==2.0.2',
        'matplotlib==3.9.2',
        'scipy==1.14.1',
        'geomdl==5.3.1',
        'geomdl.shapes==1.3.0',
        'pandas==2.2.3',
        'openpyxl==3.1.5',
        'pyevtk==1.6.0'
    ]
)