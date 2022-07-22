from setuptools import setup, find_namespace_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='astro-delight',
    version='0.0.3',    
    description='Deep Learning Identification of Galaxy Hosts in Transients, a package to automatically identify host galaxies of transient candidates',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fforster/delight',
    author='Francisco Förster',
    author_email='francisco.forster@gmail.com',
    license='GNU GPLv3',
    packages=find_namespace_packages(include=["delight.*"]),
    install_requires=['astropy',
                      'sep',
                      'xarray',
                      'panstamps',
                      'matplotlib',
                      'numpy',
                      'tensorflow'
                      ],
    build_requires=['astropy',
                      'sep',
                      'xarray',
                      'panstamps',
                      'matplotlib',
                      'numpy',
                      'tensorflow'
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
