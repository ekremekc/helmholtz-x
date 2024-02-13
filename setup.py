from setuptools import setup

setup(
    name='helmholtz_x',
    version = '0.0',
    author='Ekrem Ekici',
    author_email='ee331@cam.ac.uk',
    packages=['helmholtz_x'],
    install_requires=[
        'h5py',
        'meshio',
        'numpy',
        'matplotlib',
        'scipy',
        'geomdl',
        'geomdl.shapes',
        'pandas',
        'openpyxl'
    ]
)