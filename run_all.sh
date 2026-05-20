#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

g++ -std=c++17 -fopenmp main.cpp matrix_genrator.cpp parallel_mpi.cpp sequential.cpp -o app
./app

if [[ ! -d "graphs/.venv" ]]; then
	python3 -m venv graphs/.venv
fi

source graphs/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib
python graphs/plot.py

