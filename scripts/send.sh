#!/bin/bash
zip -r sparse-matrix-solver.zip src makefile scripts tests
scp sparse-matrix-solver.zip s279654@crescent.central.cranfield.ac.uk:sparse-matrix-solver
rm sparse-matrix-solver.zip