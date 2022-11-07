#!/bin/bash



for size in 64 128 256
do
   for version in 2
   do
      for turn in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
      do
         echo $size.$version.$turn.adhoc        
         $JULIA_PATH/julia ./run_sample.jl $version $size $turn >> output.iq.adhoc.$version.$size
         echo $size.$version.$turn.structured
         PLATFORM_DESCRIPTION=Platform.$version.toml $JULIA_PATH/julia ./run_sample.jl -$version $size $turn >> output.iq.structured.$version.$size     
      done
   done
done
