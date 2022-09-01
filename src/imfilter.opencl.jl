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
    (cl.Buffer(T, GPU.ctx, :copy, hostbuf=A), clpad, size(img), size(A), length(A))
end

struct CLKArray{T,N} <: AbstractArray{T,N}
    bufA
    bufA_
    img
    clpad
    img_size 
    A_size 
    A_length

    function CLKArray(img) 
        N = ndims(img)
        T = eltype(img)
        (bufA1, clpad, img_size, A_size, A_length) = clkarray_create(img)
        bufA2 = clkarray_create(img.^2)[1]
        new{T,N}(bufA1, bufA2, img, clpad, img_size, A_size, A_length)
    end

    function CLKArray(T, N, bufA, bufA_,img,clpad, img_size, A_size, A_length)
        new{T,N}(bufA, bufA_, img, clpad, img_size, A_size, A_length)
    end

end


function power2(a::CLKArray{T,N}) where {T,N} 
    CLKArray(T, N, a.bufA_, a.bufA, a.img, a.clpad, a.img_size, a.A_size, a.A_length) 
end

Base.size(A::CLKArray{T,N}) where {T,N} = size(A.img)

#Base.IndexStyle(::Type{<:CLKArray{T,N}}) where {T,N} = IndexCartesian()

#Base.getindex(A::CLKArray{T,N}, i::Int) where {T,N} = A.img[i]

Base.getindex(A::CLKArray{T,N}, I::Vararg{Int,N}) where {T,N} = get(A.img, I, zero(T))

#setindex!(A::CLKArray{T,N}, v, i::Int)

#setindex!(A::CLKArray{T,N}, v, I::Vararg{Int,N})

@platform aware function array_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, img) 
    CLKArray(img) 
end

@platform aware function view_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, array, I) view(array, I) end

@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, img, kern)  
    
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
 
   krn = zeros(T, img.A_size)
   indexesK = ntuple(d->[img.A_size[d]-prepad[d]+1:img.A_size[d];1:size(kern,d)-prepad[d]], N)
   krn[indexesK...] = reflect(kern) 

   # plan FFT
   p = clfft.Plan(T, ctx, img.A_size)
   clfft.set_layout!(p, :interleaved, :interleaved)
   clfft.set_result!(p, :inplace)
   clfft.bake!(p, queue)
 
   # populate GPU memory
#   bufA   = cl.Buffer(T, ctx, :copy, hostbuf=A)
   bufA_   = cl.Buffer(T, ctx, :alloc, img.A_length)
   bufkrn = cl.Buffer(T, ctx, :copy, hostbuf=krn)
   bufRES = cl.Buffer(T, ctx, img.A_length)
   
   # compute ifft(fft(A).*fft(kern))
   clfft.enqueue_transform(p, :forward, [queue], img.bufA, bufA_)
   clfft.enqueue_transform(p, :forward, [queue], bufkrn, nothing)
   queue(mult_kernel, img.A_length, nothing, bufA_, bufkrn, bufRES)

   clfft.enqueue_transform(p, :backward, [queue], bufRES, nothing)

   # get result back
   AF = reshape(cl.read(queue, bufRES), img.A_size)
 
   # undo OpenCL FFT paddings
   AF = view(AF, ntuple(d->1:size(AF,d)-img.clpad[d], N)...)
  
   out = Array{realtype(eltype(AF))}(undef, ntuple(d->img.img_size[d] - prepad[d] - postpad[d], N)...)
   indexesA = ntuple(d->postpad[d]+1:img.img_size[d]-prepad[d], N)
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


