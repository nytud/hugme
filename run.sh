#!/bin/bash

virtualenv -p python3 venv

source venv/bin/activate
venv/bin/pip3 install -r requirements.txt
pip install .
hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json
deactivate
