from setuptools import setup, find_packages

setup(
    name='chemgenpy',
    version='0.1.0',
    author='Ahmed Ogunjobi',
    author_email='aoo14@txstate.edu',
    description='ChemGenPy is a smart and massive parallel molecular library generator for chemical and materials sciences.',
    url='https://github.com/aogunjobi/ChemGenPy',
    packages=find_packages(),
    install_requires=[
        'future',
        'six',
        'numpy',
        'pandas',
        'scipy',
        'ipywidgets',
        'rdkit',  
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: BSD License',
    ],
)
