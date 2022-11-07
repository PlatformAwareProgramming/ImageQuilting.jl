import Pkg; Pkg.activate(".")
using ImageQuilting
using CUDA

function main(args)

        @info args

        v = parse(Int64,args[1])
        size = parse(Int64,args[2])
        i = parse(Int64,args[3])

        img = rand(Float32,(size,size,size))
        krn = rand(Float32,(10,10,10))

        if (v < 0)
           @info "$i: structured / $v"
           img = ImageQuilting.array_kernel(img)
           ImageQuilting.init_imfilter_kernel()
           @time ImageQuilting.imfilter_kernel(img,krn)
           ImageQuilting.imfilter_kernel(img,krn)
           ImageQuilting.imfilter_kernel(img,krn)
           @time ImageQuilting.imfilter_kernel(img,krn)
        elseif (v == 1)
           @info "$i: ad-hoc / default"
           @time ImageQuilting.imfilter_cpu(img, krn)
           @time ImageQuilting.imfilter_cpu(img, krn)
        elseif (v == 2)
           @info "$i: ad-hoc / CUDA"
           img = CuArray(img)
           @time ImageQuilting.imfilter_cuda(img, krn)
           ImageQuilting.imfilter_cuda(img, krn)
           ImageQuilting.imfilter_cuda(img, krn)
           @time ImageQuilting.imfilter_cuda(img, krn)
        elseif (v == 3)
           @info "$i: ad-hoc / OpenCL"
           ImageQuilting.init_opencl_context()
           @time ImageQuilting.imfilter_opencl(img, krn)
           ImageQuilting.imfilter_opencl(img, krn)
           ImageQuilting.imfilter_opencl(img, krn)
           @time ImageQuilting.imfilter_opencl(img, krn)
        else
           @info "wrong selection"
        end
end

main(ARGS)

