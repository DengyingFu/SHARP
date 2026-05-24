#!/usr/bin/env bash

# Move to the root directory of the repo
DIRNAME="$( dirname -- "$( readlink -f -- "$0"; )"; )"
START_PATH="$( realpath $DIRNAME/.. )"
cd $START_PATH

# Download the weights
cd "assets"
mkdir -p "weights"
cd weights
wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt