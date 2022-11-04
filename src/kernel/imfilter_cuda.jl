# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@platform aware function init_imfilter_kernel({accelerator_count::(@atleast 1), 
                                               accelerator_manufacturer::NVIDIA, 
                                               accelerator_api::(@api CUDA)})
  @info "Running on NVIDIA/CUDA GPU"
end

@platform aware array_kernel({accelerator_count::(@atleast 1), 
                              accelerator_manufacturer::NVIDIA,
                              accelerator_api::(@api CUDA)}, array) = CuArray(array)

@platform aware view_kernel({accelerator_count::(@atleast 1), 
                             accelerator_manufacturer::NVIDIA, 
                             accelerator_api::(@api CUDA)}, array, I)  = Array(array[I])

@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), 
                                          accelerator_manufacturer::NVIDIA, 
                                          accelerator_api::(@api CUDA)}, img, krn)
   imfilter_cuda(img,krn)
end

function imfilter_cuda(img, krn)
 
   # pad kernel to common size with image
   padkrn = CUDA.zeros(size(img))
   copyto!(padkrn, CartesianIndices(krn), CuArray(krn), CartesianIndices(krn))
 
   # perform ifft(fft(img) .* conj.(fft(krn)))
   fftimg = img |> CUFFT.fft
   fftkrn = padkrn |> CuArray |> CUFFT.fft
   result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft
 
   # recover result
   finalsize = size(img) .- (size(krn) .- 1)
   real.(result[CartesianIndices(finalsize)]) |> Array
end
