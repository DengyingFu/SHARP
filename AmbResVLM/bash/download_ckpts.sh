#!/usr/bin/env bash

# Move to the root directory of the repo
DIRNAME="$( dirname -- "$( readlink -f -- "$0"; )"; )"
START_PATH="$( realpath $DIRNAME/.. )"
cd $START_PATH

# Download the weights
wget -q --show-progress https://ambres.cs.uni-freiburg.de/download/ckpt.zip
unzip ckpt.zip
rm ckpt.zip