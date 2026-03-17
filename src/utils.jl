module Utils

using CUDA
using Flux
using cuDNN

export log_message, gpu_available, to_device, to_cpu

function log_message(message::AbstractString)
    println(message)
end

gpu_available() = CUDA.functional()

to_device(x) = gpu_available() ? CUDA.cu(x) : x

to_cpu(x) = Flux.cpu(x)

end
