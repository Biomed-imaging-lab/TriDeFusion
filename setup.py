from setuptools import setup, find_packages
from os import path


_dir = path.abspath(path.dirname(__file__))


with open(path.join(_dir,'tridefusion','version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()


setup(name='tridefusion',
      version=__version__,
      description='TriDeFusion - a toolbox for Content-aware Image Restoration (CARE)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://csbdeep.bioimagecomputing.com/',
      author='Uwe Schmidt, Martin Weigert',
      author_email='research@uweschmidt.org, martin.weigert@epfl.ch',
      license='Appache 2.0 License',
      packages=find_packages(),
      python_requires='>=3.10',

      project_urls={
          'Documentation': 'http://csbdeep.bioimagecomputing.com/doc/',
          'Repository': 'https://github.com/csbdeep/csbdeep',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
      ],

      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "six",
          "tifffile",
          "tqdm",
          "packaging",
      ],

      extras_require={
          "tf1":  ["keras>=2.1.2,<2.4","protobuf<3.21","h5py<3"],
          "test": ["pytest"],
          "docs": ["sphinx","sphinx-rtd-theme"],
      },

      entry_points={
          'console_scripts': [
              'care_predict = csbdeep.scripts.care_predict:main'
          ]
      }
      )