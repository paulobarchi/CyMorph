# CyMorph
### Non-parametric Galaxy Morphology Package

Adapted and improved from original CyMorph:
https://github.com/rsautter/CyMorph

## Requirements
 - Python 2.7
 - MPI / MPI4py
 - Astropy
 - Numpy
 - SciPy
 - Cython

## Compiling

    python compile.py build_ext --inplace
    
### Paths
There should be specified in 'cfg/paths.ini', where is/how to call R, Python, and SExtractor
 
## Running example
Single-core and single object (without clipping image):

    time python main.py config.ini
    
MPI run (clipping the image):

    mpirun -np 3 PCyMorph.sh test500/spirals.csv
    
## Configure File
In order to run, a config file is required (in the example the config.ini where used). To run with MPI support, the default configuration file is ParallelConfig.ini.
This configuration file contain information about files, the input/output, and the parameters.
An example is:

    [File_Configuration]
    indexes: C, H, A3, S3, Ga, OGa
    cleanit: False
    download: True
    stamp_size: 5

    [Output_Configuration]
    verbose: False
    savefigure: False

    [Indexes_Configuration]
    Entropy_Bins: 180
    Ga_Tolerance: 0.02
    Ga_Angular_Tolerance: 0.02
    Ga_Position_Tolerance: 0.00
    Concentration_Density: 100
    Concentration_Distances: 0.65, 0.25
    butterworth_order: 2
    smooth_degree: 0.2

## Extras

### Downloading from SDSS
To download images from SDSS (without executing the pipeline), a module in Download was created (it requires a .csv file)
In order to run it:

    mpirun -np 2 python downloader.py test500/spirals.csv
    
### Parametrization
The file 'optimizeIndexesThreshold.py' is a framework for optimal parametrization, it works as a wrapper to the pipeline.
Beyond indexes parametrization, it also test sExtractor detection threshold.
It measures the distribution distance/divergence between two known sets. 
It requires the data in folder 'Field/'.

