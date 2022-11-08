# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@platform default function init_imfilter_kernel()
  @info "Running default kernel"
end

@platform default array_kernel(array) = array

@platform default view_kernel(array, I) = view(array, I)

@platform default function imfilter_kernel(img, krn)
  imfilter_cpu(img, krn)
end

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end
