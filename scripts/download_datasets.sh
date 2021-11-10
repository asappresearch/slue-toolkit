#!/bin/bash

#1. Download
aws-okta exec asapp-dev-okta-speech-systems -- aws s3 cp s3://asapp-lang-tech/users/sshon/slue_first_phase/slue-voxceleb_blind.tar.gz datasets/
aws-okta exec asapp-dev-okta-speech-systems -- aws s3 cp s3://asapp-lang-tech/users/sshon/slue_first_phase/slue-voxpopuli_blind.tar.gz datasets/

#2. Extract
tar -xzvf datasets/slue-voxceleb_blind.tar.gz -C datasets/
tar -xzvf datasets/slue-voxpopuli_blind.tar.gz -C datasets/

#3. preprocess

python slue_toolkit/prepare/prepare_voxceleb.py create_manifest
python slue_toolkit/prepare/prepare_voxpopuli.py create_manifest
