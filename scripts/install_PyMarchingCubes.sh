git clone https://github.com/JustusThies/PyMarchingCubes.git
cd PyMarchingCubes
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout tags/3.4.0
cd ..
python setup.py install
cd .. 