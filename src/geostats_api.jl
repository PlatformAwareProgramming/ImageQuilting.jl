## Copyright (c) 2017, Júlio Hoffimann Mendes <juliohm@stanford.edu>
##
## Permission to use, copy, modify, and/or distribute this software for any
## purpose with or without fee is hereby granted, provided that the above
## copyright notice and this permission notice appear in all copies.
##
## THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
## WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
## ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
## WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
## ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
## OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
    ImgQuilt(var₁=>param₁, var₂=>param₂, ...)

Image quilting simulation solver.

## Parameters

* `TI` - Training image
* `template` - Template size in x, y and z
* `overlap`  - Overlap size in x, y and z (default to (1/6, 1/6, 1/6))
* `soft`     - An instance of `SoftData`
* `cut`      - Boundary cut algorithm (:boykov (default) or :dijkstra)
* `path`     - Simulation path (:rasterup (default), :rasterdown, :dilation, or :random)
* `simplex`  - Whether to apply or not the simplex transform
* `tol`      - Initial relaxation tolerance in (0,1] (default to 0.1)

## Global parameters

* `threads`      - Number of threads in FFT (default to number of physical CPU cores)
* `gpu`          - Whether to use the GPU or the CPU (default to false)
* `showprogress` - Whether to show or not the estimated time duration (default to false)
"""
@simsolver ImgQuilt begin
  @param TI
  @param template
  @param overlap       = (1/6, 1/6, 1/6)
  @param soft          = nothing
  @param cut           = :boykov
  @param path          = :rasterup
  @param simplex       = false
  @param tol           = .1
  @global threads      = CPU_PHYSICAL_CORES
  @global gpu          = false
  @global showprogress = false
end

function solve_single(problem::SimulationProblem, var::Symbol, solver::ImgQuilt)
  # retrieve problem info
  pdata = data(problem)
  pdomain = domain(problem)

  # sanity checks
  @assert pdomain isa RegularGrid "ImgQuilt solver only supports regular grids"
  @assert ndims(pdomain) ∈ [2,3] "Number of dimensions must be 2 or 3"

  # get user parameters
  if var in keys(solver.params)
    varparams = solver.params[var]
  else
    varparams = ImgQuiltParam()
  end

  # add ghost dimension to simulation grid if necessary
  sz = ndims(pdomain) == 2 ? (size(pdomain)..., 1) : size(pdomain)

  # create hard data object
  hd = HardData()
  for (loc, datloc) in datamap(problem, var)
    push!(hd, ind2sub(sz, loc) => value(pdata, datloc, var))
  end

  # run image quilting core function
  reals = iqsim(varparams.TI, varparams.template..., sz...,
                soft=varparams.soft, hard=hd, tol=varparams.tol,
                cut=varparams.cut, path=varparams.path, simplex=varparams.simplex,
                threads=solver.threads, gpu=solver.gpu, showprogress=solver.showprogress)

  # return result
  reals[1][:]
end