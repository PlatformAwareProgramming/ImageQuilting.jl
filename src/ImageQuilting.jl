# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using Meshes
using GeoStatsBase

using Tables
using Graphs
using GraphsFlows
using ImageFiltering
using ImageMorphology
using StatsBase: sample, weights
using ProgressMeter: Progress, next!
using FFTW: set_num_threads
using CpuId: cpucores
using RecipesBase

using CUDA
using OpenCL
using CLFFT
using PlatformAware

using Base: @nexprs, @nloops, @nref
using SparseArrays: spzeros
using Statistics: mean, std
using Random
 
import GeoStatsBase: preprocess, solvesingle

include("utils.jl")
include("plotrecipes.jl")
include("relaxation.jl")
include("taumodel.jl")
include("graphcut.jl")
include("iqsim.jl")
include("voxelreuse.jl")
include("geostats.jl")

function __init__()    
  include("src/kernels.jl")
  include("src/kernel/imfilter_default.jl")
  include("src/kernel/imfilter_cuda.jl")
  include("src/kernel/imfilter_opencl.jl")
end

include("test_imfilter.jl")

export
  # functions
  iqsim,
  voxelreuse,

  # geostats solver
  IQ

end
