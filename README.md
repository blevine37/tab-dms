# TAB-DMS
Fully time-dependent non-adiabatic dynamics software that uses Terachem's TDCI as an electronic dynamics driver. Currently can do Ehrenfest dynamics, but Collapse-To-A-Block in Dense Manifolds of States (TAB-DMS) method coming soon.


## Installation
Requires: python2, numpy, h5py, and the 'tdci\_testing' build of Terachem
To install h5py for python2 on ubuntu:
`apt install libhdf5-dev`
`HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/`
`pip2 install h5py`

To install tab-dms, simply symlink \__main__.py to a $PATH directory
`ln -s /home/adurden/tab-dms/__main__.py /home/adurden/bin/tab`


## Running tab-dms
Create a new directory, copy the initial geometry (in .xyz file format) and inputfile.py to the directory.
Modify inputfile.py to your liking, and then simply run
`tab`
alternatively you can manually run
`python2 /path/to/tab-dms/__main__.py`





