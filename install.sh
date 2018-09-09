#!/bin/bash

cd dnn/extensions
mkdir build
cd build
cmake ../
make
cd ..
python setup.py build
