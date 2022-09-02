# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@platform aware function init_imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API})
    println("Running on OpenCL GPU")
    global GPU = gpu_setup()
  end
  
  T = ComplexF64
  
  
  function clkarray_create(img)
      img = T.(img)
  
      # OpenCL FFT expects powers of 2, 3, 5, 7, 11 or 13
      clpad = clfftpad(img) 
      A = padarray(img, Pad(:symmetric, zeros(Int, ndims(img)), clpad))
      A = parent(A)
  
      # populate GPU memory
      (cl.Buffer(T, GPU.ctx, :copy, hostbuf=A), A, clpad)
  end
  
  struct CLKArray{T,N} <: AbstractArray{T,N}
      bufA    # img 
      bufA_   # img^2
      img
      A
      clpad
      
      function CLKArray(img) 
        N = ndims(img)
        T = eltype(img)
        (bufA1, A, clpad) = clkarray_create(img)
        bufA2 = clkarray_create(img.^2)[1]
        new{T,N}(bufA1, bufA2, img, A, clpad)
      end
  
      function CLKArray(T, N, bufA, bufA_,img, A, clpad)
        new{T,N}(bufA, bufA_, img, A, clpad)
      end
  
  end  
  
  power2(a::CLKArray{T,N}) where {T,N} = CLKArray(T, N, a.bufA_, a.bufA, a.img, a.A, a.clpad) 
  
  Base.size(A::CLKArray{T,N}) where {T,N}= size(A.img)
  
  Base.getindex(A::CLKArray{T,N}, I::Vararg{Int,N}) where {T,N} = get(A.img, I, zero(T))
    
  @platform aware function array_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, img) 
      CLKArray(img) 
  end
  
  @platform aware function view_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, array, I) view(array, I) end
  
  @platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, img, kern)  
      imfilter_opencl(img, kern)
  end
  
  
  function imfilter_opencl(img, kern)

    # retrieve basic info
     N = ndims(img.img)
     #T = ComplexF64
  
     # GPU metadata
     ctx = GPU.ctx; queue = GPU.queue
     mult_kernel = GPU.mult_kernel
   
     # operations with complex type
  #   img  = T.(img)
     kern = T.(kern)
     
     # kernel may require padding
     prepad  = ntuple(d->(size(kern,d)-1) ÷ 2, N)
     postpad = ntuple(d->(size(kern,d)  ) ÷ 2, N)
   
     # OpenCL FFT expects powers of 2, 3, 5, 7, 11 or 13
  #   clpad = clfftpad(img)
  #   A = padarray(img, Pad(:symmetric, zeros(Int, ndims(img)), clpad))
  #   A = parent(A)
   
     krn = zeros(T, size(img.A))
     indexesK = ntuple(d->[size(img.A,d)-prepad[d]+1:size(img.A,d);1:size(kern,d)-prepad[d]], N)
     krn[indexesK...] = reflect(kern) 
  
     # plan FFT
     p = clfft.Plan(T, ctx, size(img.A))
     clfft.set_layout!(p, :interleaved, :interleaved)
     clfft.set_result!(p, :inplace)
     clfft.bake!(p, queue)
  
     p_ = clfft.Plan(T, ctx, size(img.A))
     clfft.set_layout!(p_, :interleaved, :interleaved)
     clfft.set_result!(p_, :outofplace)
     clfft.bake!(p_, queue)
  
     # populate GPU memory
  #   bufA   = cl.Buffer(T, ctx, :copy, hostbuf=A)
     bufA_   = cl.Buffer(T, ctx, :alloc, length(img.A))
     bufkrn = cl.Buffer(T, ctx, :copy, hostbuf=krn)
     bufRES = cl.Buffer(T, ctx, length(img.A))
     
     # compute ifft(fft(A).*fft(kern))
     clfft.enqueue_transform(p_, :forward, [queue], img.bufA, bufA_)
     clfft.enqueue_transform(p, :forward, [queue], bufkrn, nothing)
     queue(mult_kernel, length(img.A), nothing, bufA_, bufkrn, bufRES)
  
     clfft.enqueue_transform(p, :backward, [queue], bufRES, nothing)
  
     # get result back
     AF = reshape(cl.read(queue, bufRES), size(img.A))
   
     # undo OpenCL FFT paddings
     AF = view(AF, ntuple(d->1:size(AF,d)-img.clpad[d], N)...)
    
     out = Array{realtype(eltype(AF))}(undef, ntuple(d->size(img,d) - prepad[d] - postpad[d], N)...)
     indexesA = ntuple(d->postpad[d]+1:size(img,d)-prepad[d], N)
     copyreal!(out, AF, indexesA)
     
    # if (Sys.free_memory() / 2^20 < 1000) 
    #  @info "GC !"
    #  GC.gc() 
    # end
  
     out
  end
  
   @generated function reflect(A::AbstractArray{T,N}) where {T,N}
      quote
          B = Array{T,N}(undef,size(A))
          @nexprs $N d->(n_d = size(A, d)+1)
          @nloops $N i A d->(j_d = n_d - i_d) begin
              @nref($N, B, j) = @nref($N, A, i)
          end
          B
      end
   end
  
  
  for N = 1:5
      @eval begin
          function copyreal!(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}}) where {T<:Real}
              @nexprs $N d->(I_d = I[d])
              @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                  (@nref $N dst i) = real(@nref $N src j)
              end
              dst
          end
          function copyreal!(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}}) where {T<:Complex}
              @nexprs $N d->I_d = I[d]
              @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                  (@nref $N dst i) = @nref $N src j
              end
              dst
          end
      end
  end
  
  realtype(::Type{R}) where {R<:Real} = R
  realtype(::Type{Complex{R}}) where {R<:Real} = R
  
  
  