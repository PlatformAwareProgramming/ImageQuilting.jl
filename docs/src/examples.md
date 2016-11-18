Consider installing the [GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package.

# Unconditional

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")
reals = iqsim(TI, 62, 62, 1, size(TI)..., nreal=3)

TI = training_image("StoneWall")
reals, cuts, voxs = iqsim(TI, 13, 13, 1, size(TI)..., nreal=3, debug=true)
```
![Unconditional simulation](images/unconditional.png)

# Hard data

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")

data = HardData()
push!(data, (50,50,1)=>1)
push!(data, (190,50,1)=>0)
push!(data, (150,170,1)=>1)
push!(data, (150,190,1)=>1)

reals, cuts, voxs = iqsim(TI, 30, 30, 1, size(TI)..., hard=data, debug=true)
```
![Hard data conditioning](images/hard.gif)

![Hard data conditioning](images/hard.png)

# Soft data

```julia
using ImageQuilting
using GeoStatsImages
using Images: imfilter_gaussian

TI = training_image("WalkerLake")
truth = training_image("WalkerLakeTruth")

G(m) = imfilter_gaussian(m, [10,10,0])

data = SoftData(G(truth), G)

reals = iqsim(TI, 27, 27, 1, size(truth)..., soft=data, nreal=3)
```
![Soft data conditioning](images/soft.png)