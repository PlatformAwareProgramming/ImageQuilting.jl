# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function init_kernel(::CUDAMethod) 
   @info "Running CUDA kernel"
 end
 
 const array_kernel(::CUDAMethod, array) = CuArray{Float32}(array)
 
 const view_kernel(::CUDAMethod, array, I) = Array(array[I])
 
 function imfilter_kernel(::CUDAMethod, img, krn)
   imfilter_cuda(img, krn)
 end
 
 function imfilter_cuda(img, krn)
 
   # pad kernel to common size with image
   padkrn = CUDA.zeros(Float32, size(img))
   copyto!(padkrn, CartesianIndices(krn), CuArray{Float32}(krn), CartesianIndices(krn))
 
   # perform ifft(fft(img) .* conj.(fft(krn)))
   fftimg = img |> CUFFT.fft
   fftkrn = padkrn |> CUFFT.fft
   result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft
 
   # recover result
   finalsize = size(img) .- (size(krn) .- 1)
   real.(result[CartesianIndices(finalsize)]) |> Array
 end
 