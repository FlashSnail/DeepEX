#!/bin/bash

#publish zzhfun to pypi
rm -rf build/ dist/ zzhfun.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/*
