#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/..
python code/run_tictactoe.py --iters 2000 --c 1.414
python -m src.run_tictactoe --iters 2000 --c 1.414
