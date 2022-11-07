#!/bin/bash

for turn in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
do
   echo $turn.adhoc
   git switch iq-sim-refact-adhoc
   PATH_IQ="." $JULIA_PATH/julia ./run_sample_app.jl $turn >> output.iq.app.adhoc
   echo $turn.structured
   git switch iq-sim-refact
   PATH_IQ="." $JULIA_PATH/julia ./run_sample_app.jl $turn >> output.iq.app.structured
done
