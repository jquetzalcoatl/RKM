

include("../grad/tikhonov_rand.jl")


s=(1,500_000)
@time α, sGL, gammaC, sGU, mT, A, p = CuArray(rand(prod(s))), CuArray(zeros(prod(s))),
        CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64,prod(s),8)),
        CuArray(ones(Float64, 8,3,prod(s))), CuArray(rand(3,prod(s)));

@time psi = TikhonovGenGPUv2(α, sGL, gammaC, sGU, mT, A);

@time p = TikhonovGenGPUv2_A(α, sGL, gammaC, sGU, mT, A);
@time psi = TikhonovGenGPUv2_B(α, sGL, gammaC, sGU, p);