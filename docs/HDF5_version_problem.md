Problem: Matlab uses hdf5 version 1.8.12. Default ubuntu hdf5 version 1.10.0

Solution:

## install hdf5 1.8.12

- Download hdf5-1.8.12 from [here](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.12/obtain51812.html)

- tar -xvf hdf5-1.8.12.tar

- cd hdf5-1.8.12

- ./configure --prefix=/usr/local/hdf5

- make -j8

- make check -j8

- sudo make install -j8

- sudo make check-install -j8 


## build h5py with hdf5=1.8.12

- [reference](http://docs.h5py.org/en/stable/build.html)

- git clone https://github.com/h5py/h5py.git

- python setup.py configure --hdf5=/usr/local/hdf5

- conda activate env_name

- pip install -e .

## test

- matlab -nodesktop -nodisplay

- py.importlib.import_module("h5py")
