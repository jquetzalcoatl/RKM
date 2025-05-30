

include("../grad/tikhonov_rand.jl")


s=(1,500_000)
CUDA.@time α, sGL, gammaC, sGU, mT, A, p = CuArray(rand(prod(s))), CuArray(zeros(prod(s))),
        CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64,prod(s),8)),
        CuArray(ones(Float64, 8,3,prod(s))), CuArray(rand(3,prod(s)));

# CUDA.@time α, sGL, gammaC, sGU, mT, A, p = CuArray(ones(prod(s))), CuArray(zeros(prod(s))),
        # CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64,prod(s),8)),
        # CuArray(ones(Float64, 8,3,prod(s))), CuArray(rand(3,prod(s)));

CUDA.@time psi = TikhonovGenGPUv2(α, sGL, gammaC, sGU, mT, A);

CUDA.@time p = TikhonovGenGPUv2_A(α, sGL, gammaC, sGU, mT, A);
CUDA.@time psi = TikhonovGenGPUv2_B(α, sGL, gammaC, sGU, p);

CUDA.@time CUDA.@sync @cuda tikhonov_loop(α, sGL, gammaC, sGU, mT, A)

CUDA.@time CUDA.@sync A_inv = batched_pinv(A);

############
using Plots

plot(Array(psi), st=:histogram)
