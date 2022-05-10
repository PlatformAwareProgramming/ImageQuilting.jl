# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_resource(resource::CUDALibs, img, kern)

  sizeall = size(img) .+ size(kern) .- 1

  imggpu = nothing
  kerngpu = nothing

  N = ndims(img)
  T = eltype(img)

  @sync begin
    Threads.@spawn begin
      imggpu = CUDA.zeros(T,sizeall)
      imgindexes = ntuple(d->let padsize = div(size(kern,d)-1,2); (padsize+1):size(img,d)+padsize end, N)
      imggpu[imgindexes...] = img
    end

    kerngpu = CUDA.zeros(T,sizeall)
    kernindexes = axes(kern)
    kerngpu[kernindexes...] = kern
    kerngpu = CUFFT.fft(kerngpu)
    kerngpu = conj.(kerngpu)
  end

  imggpu = CUFFT.fft(imggpu)

  out = imggpu .* kerngpu

  out = CUFFT.ifft(out)
 
  # remove padding
  ix = map(x -> Base.OneTo(x), size(img) .- size(kern) .+ 1)
  out = out[ix...]

  out = real.(out)
  out = Array(out)

  out
end