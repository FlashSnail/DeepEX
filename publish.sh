#!/bin/bash

#publish deepex to pypi
rm -rf build/ dist/ deepex.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/*
