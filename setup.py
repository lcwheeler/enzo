import sys
import numpy

if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

# Try using setuptools first, if it's installed
from setuptools import setup, find_packages


# Need to add all dependencies to setup as we go!
setup(name='enzo',
      packages=find_packages(),
      version='0.1',
      description="Python software package for evolutionary simulations of enzyme pathways",
      author='Lucas C. Wheeler',
      author_email='lwheeler9@gmail.com',
      zip_safe=False,
      install_requires=["matplotlib","numpy","pandas", "tellurium", "seaborn"],
      classifiers=['Programming Language :: Python'])
