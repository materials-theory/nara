import io
from setuptools import find_packages, setup, Extension
from nara.firefly.main import __version__

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(name = 'nara-fa',
      version = __version__,
      description='',
      long_description=long_description(),
      url='https://github.com/materials-theory/nara',
      author='Giyeok Lee',
      author_email='giyeok.lee@sydney.edu.au',
      license='MIT',
      packages=find_packages(),
      include_package_data = True,
      zip_safe = False,
      keywords='Firefly-Algorithm GNN DFT ASE Global-Optimization',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          ],
      install_requires=['ase', 'numpy', 'torch', 'e3nn', 'scikit-learn', 'scipy', 'spglib', 'llumys'],
    #   entry_points = {'console_scripts':['nara = nara.firefly.main:main']}
      )