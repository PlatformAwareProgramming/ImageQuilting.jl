# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    iqsim(trainimg::AbstractArray{T,N}, tilesize::Dims{N},
          simsize::Dims{N}=size(trainimg);
          overlap::NTuple{N,<:Real}=ntuple(i->1/6,N),
          soft::AbstractVector=[], hard::Dict=Dict(), tol::Real=.1,
          path::Symbol=:raster, nreal::Integer=1,
          threads::Integer=cpucores(), gpu::Bool=false,
          debug::Bool=false, showprogress::Bool=false)

Performs image quilting simulation as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` is any 3D array (add ghost dimension for 2D)
* `tilesize` is the tile size (or pattern size)

### Optional

* `simsize` is the size of the simulation grid (default to training image size)
* `overlap` is the percentage of overlap (default to 1/6 of tile size)
* `soft` is a vector of `(data,dataTI)` pairs (default to none)
* `hard` is a dictionary mapping coordinates to data values (default to none)
* `tol` is the initial relaxation tolerance in (0,1] (default to .1)
* `path` is the simulation path (`:raster`, `:dilation` or `:random`)
* `nreal` is the number of realizations (default to 1)
* `threads` is the number of threads for the FFT (default to all CPU cores)
* `gpu` informs whether to use the GPU or the CPU (default to false)
* `debug` informs whether to export or not the boundary cuts and voxel reuse
* `showprogress` informs whether to show or not estimated time duration

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.
"""
function iqsim(trainimg::AbstractArray{T,N}, tilesize::Dims{N},
               simsize::Dims{N}=size(trainimg);
               overlap::NTuple{N,<:Real}=ntuple(i->1/6,N),
               soft::AbstractVector=[], hard::Dict=Dict(), tol::Real=.1,
               path::Symbol=:raster, nreal::Integer=1,
               threads::Integer=cpucores(), gpu::Bool=false,
               debug::Bool=false, showprogress::Bool=false) where {T,N}

  # number of threads in FFTW
  set_num_threads(threads)

  # sanity checks
  @assert ndims(trainimg) == 3 "image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< tilesize .≤ size(trainimg)) "invalid tile size"
  @assert all(simsize .≥ tilesize) "invalid grid size"
  @assert all(0 .< overlap .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert path ∈ [:raster,:dilation,:random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # soft data checks
  if !isempty(soft)
    for (aux, auxTI) in soft
      @assert ndims(aux) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all(size(aux) .≥ simsize) "soft data size < grid size"
      @assert size(auxTI) == size(trainimg) "auxiliary TI must have the same size as TI"
    end
  end

  # hard data checks
  if !isempty(hard)
    coords = [coord[i] for i in 1:N, coord in keys(hard)]
    @assert all(maximum(coords, dims=2) .≤ simsize) "hard data coordinates outside of grid"
    @assert all(minimum(coords, dims=2) .> 0) "hard data coordinates must be positive indices"
  end

  # calculate the overlap size from given percentage
  ovlsize = @. ceil(Int, overlap*tilesize)

  # warn in case of 1-voxel overlaps
  if any((tilesize .>  1) .& (ovlsize .== 1))
    @warn "Overlaps with only 1 voxel, check tilesize/overlap configuration"
  end

  # spacing in raster path
  spacing = @. tilesize - ovlsize

  # calculate the number of tiles from grid size
  ntiles = @. ceil(Int, simsize / max(spacing, 1))

  # padded simulation grid size
  padsize = @. ntiles*(tilesize - ovlsize) + ovlsize

  # training image dimensions
  TIsize = size(trainimg)

  # distance matrix size
  distsize = TIsize .- tilesize .+ 1

  # total overlap volume in simulation grid
  ovlvol = prod(padsize) - prod(@. padsize - (ntiles - 1)*ovlsize)

  # geometric configuration
  geoconfig = (ntiles=ntiles, tilesize=tilesize, ovlsize=ovlsize, spacing=spacing,
               TIsize=TIsize, simsize=simsize, padsize=padsize, distsize=distsize)

  # pad input images and knockout inactive voxels
  TI, SOFT = preprocess_images(trainimg, soft, geoconfig)

  # disable tiles in the training image if they contain inactive voxels
  disabled = find_disabled(trainimg, geoconfig)

  # determine tiles that should be skipped and tiles with data
  skipped, datainds = find_skipped(hard, geoconfig)

  # construct simulation path
  simpath = genpath(ntiles, path, datainds)

  # show progress and estimated time duration
  showprogress && (progress = Progress(nreal))

  # main output is a vector of grids
  realizations = Vector{Array{Float64,N}}()

  # for each realization we have:
  boundarycuts = Vector{Array{Float64,N}}()
  voxelreuse   = Vector{Float64}()

  # preallocate memory
  cutmask   = Array{Bool}(undef, tilesize)
  ovldist   = Array{Float64}(undef, distsize)
  softdists = [Array{Float64}(undef, distsize) for i in 1:length(soft)]
  if !isempty(hard)
    hardmask = Array{Bool}(undef, tilesize)
    harddev  = Array{Float64}(undef, tilesize)
    harddist = Array{Float64}(undef, distsize)
  end

  for real in 1:nreal
    # allocate memory for current simulation
    simgrid = zeros(padsize)
    debug && (cutgrid = zeros(padsize))

    # keep track of pasted tiles
    pasted = Set{CartesianIndex{N}}()

    # loop simulation grid tile by tile
    for ind in simpath
      # skip tile if all voxels are inactive
      ind ∈ skipped && continue

      # Cartesian index of tile
      tileind = lin2cart(ntiles, ind)

      # tile corners are given by start and finish
      start  = @. (tileind.I - 1)*spacing + 1
      finish = @. start + tilesize - 1
      tile   = CartesianIndex(start):CartesianIndex(finish)

      # current simulation dataevent
      simdev = view(simgrid, tile)

      # overlap distance
      overlap_distance!(ovldist, TI, simdev, tileind, pasted, geoconfig)
      ovldist[disabled] .= Inf

      # hard distance
      hardtile = false
      if !isempty(hard)
        indicator!(hardmask, hard, tile)
        if any(hardmask)
          event!(harddev, hard, tile)
          hard_distance!(harddist, TI, harddev, hardmask)
          harddist[disabled] .= Inf
          hardtile = true
        end
      end

      # soft distance
      for s in eachindex(SOFT)
        AUX, AUXTI = SOFT[s]
        softdev = view(AUX, tile)
        soft_distance!(softdists[s], AUXTI, softdev)
        softdists[s][disabled] .= Inf
      end

      # main and auxiliary distances
      if hardtile
        D, Ds = harddist, [ovldist, softdists...]
      else
        D, Ds = ovldist, softdists
      end

      # current pattern database
      patterndb = isempty(Ds) ? findall(vec(D .≤ (1+tol)minimum(D))) : relaxation(D, Ds, tol)

      # pattern probability
      patternprobs = tau_model(patterndb, D, Ds)

      # pick a pattern at random from the database
      rind   = sample(patterndb, weights(patternprobs))
      start  = lin2cart(size(D), rind)
      finish = @. start.I + tilesize - 1
      rtile  = CartesianIndex(start):CartesianIndex(finish)

      # selected training image dataevent
      TIdev = view(TI, rtile)

      # boundary cut mask
      cutmask .= false
      for d=1:N
        # Cartesian index of previous and next tiles along dimension
        prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d]-1) : tileind[i], N))
        next = CartesianIndex(ntuple(i -> i == d ? (tileind[d]+1) : tileind[i], N))

        # compute mask with previous tile
        if ovlsize[d] > 1 && prev ∈ pasted
          oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds); B = view(TIdev, inds)
          cutmask[inds] .|= boykov_kolmogorov_cut(A, B, d)
        end

        # compute mask with next tile
        if ovlsize[d] > 1 && next ∈ pasted
          oslice = ntuple(i -> i == d ? (spacing[d]+1:tilesize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds); B = view(TIdev, inds)
          cutmask[inds] .|= reverse(boykov_kolmogorov_cut(reverse(A, dims=d), reverse(B, dims=d), d), dims=d)
        end
      end

      # paste quilted pattern from training image
      simdev[.!cutmask] = TIdev[.!cutmask]

      # save boundary cut
      debug && (cutgrid[tile] = cutmask)

      # mark tile as pasted
      push!(pasted, tileind)
    end

    # save voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/ovlvol)

    # hard data and shape correction
    if !isempty(hard)
      for (coord, val) in hard
        simgrid[coord] = val
      end
      if debug
        for (coord, val) in hard
          isnan(val) && (cutgrid[coord] = val)
        end
      end
    end

    # throw away voxels that are outside of the grid
    simgrid = view(simgrid, CartesianIndices(simsize))
    debug && (cutgrid = view(cutgrid, CartesianIndices(simsize)))

    # save and continue
    push!(realizations, simgrid)
    debug && push!(boundarycuts, cutgrid)

    # update progress bar
    showprogress && next!(progress)
  end

  debug ? (realizations, boundarycuts, voxelreuse) : realizations
end

function preprocess_images(trainimg::AbstractArray{T,N}, soft::AbstractVector,
                           geoconfig::NamedTuple) where {T,N}
  padsize = geoconfig.padsize

  TI = Float64.(trainimg)
  replace!(TI, NaN => 0.)

  SOFT = map(soft) do (aux, auxTI)
    prepend = ntuple(i->0, N)
    append  = padsize .- min.(padsize, size(aux))
    padding = Pad(:symmetric, prepend, append)

    AUX   = Float64.(padarray(aux, padding))
    AUXTI = Float64.(auxTI)
    replace!(AUX, NaN => 0.)
    replace!(AUXTI, NaN => 0.)

    AUX, AUXTI
  end

  TI, SOFT
end

function find_disabled(trainimg::AbstractArray{T,N}, geoconfig::NamedTuple) where {T,N}
  TIsize = geoconfig.TIsize
  tilesize = geoconfig.tilesize
  distsize = geoconfig.distsize

  disabled = falses(distsize)
  for ind in findall(isnan, trainimg)
    start  = @. max(ind.I - tilesize + 1, 1)
    finish = @. min(ind.I, distsize)
    tile   = CartesianIndex(start):CartesianIndex(finish)
    disabled[tile] .= true
  end

  disabled
end

function find_skipped(hard::Dict, geoconfig::NamedTuple)
  ntiles = geoconfig.ntiles
  tilesize = geoconfig.tilesize
  spacing = geoconfig.spacing
  simsize = geoconfig.simsize

  skipped = Set{Int}()
  datainds = Vector{Int}()
  for tileind in CartesianIndices(ntiles)
    # tile corners are given by start and finish
    start  = @. (tileind.I - 1)*spacing + 1
    finish = @. start + tilesize - 1
    tile   = CartesianIndex(start):CartesianIndex(finish)

    # skip tile if either
    #   1) tile is beyond true simulation size
    #   2) all values in the tile are NaN
    if any(start .> simsize) || !any(activation(hard, tile))
      push!(skipped, cart2lin(ntiles, tileind))
    elseif any(indicator(hard, tile))
      push!(datainds, cart2lin(ntiles, tileind))
    end
  end

  skipped, datainds
end

function overlap_distance!(distance::AbstractArray{T,N},
                           TI::AbstractArray{T,N}, simdev::AbstractArray{T,N},
                           tileind::CartesianIndex{N}, pasted::Set{CartesianIndex{N}},
                           geoconfig::NamedTuple) where {N,T<:Real}
  TIsize = geoconfig.TIsize
  tilesize = geoconfig.tilesize
  ovlsize = geoconfig.ovlsize
  spacing = geoconfig.spacing

  distance .= 0
  for d=1:N
    # Cartesian index of previous and next tiles along dimension
    prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d]-1) : tileind[i], N))
    next = CartesianIndex(ntuple(i -> i == d ? (tileind[d]+1) : tileind[i], N))

    # compute overlap distance with previous tile
    if ovlsize[d] > 1 && prev ∈ pasted
      oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
      ovl = view(simdev, CartesianIndices(oslice))

      D = convdist(TI, ovl)

      ax = axes(D)
      dslice = ntuple(i -> i == d ? (1:TIsize[d]-tilesize[d]+1) : ax[i], N)
      distance .+= view(D, CartesianIndices(dslice))
    end

    # compute overlap distance with next tile
    if ovlsize[d] > 1 && next ∈ pasted
      oslice = ntuple(i -> i == d ? (spacing[d]+1:tilesize[d]) : (1:tilesize[i]), N)
      ovl = view(simdev, CartesianIndices(oslice))

      D = convdist(TI, ovl)

      ax = axes(D)
      dslice = ntuple(i -> i == d ? (spacing[d]+1:TIsize[d]-ovlsize[d]+1) : ax[i], N)
      distance .+= view(D, CartesianIndices(dslice))
    end
  end
end

function hard_distance!(distance::AbstractArray{T,N},
                        TI::AbstractArray{T,N},
                        harddev::AbstractArray{T,N},
                        hardmask::AbstractArray{Bool,N}) where {N,T<:Real}
  distance .= convdist(TI, harddev, weights=hardmask)
end

function soft_distance!(distance::AbstractArray{T,N},
                        TI::AbstractArray{T,N},
                        softdev::AbstractArray{T,N}) where {N,T<:Real}
  distance .= convdist(TI, softdev)
end
