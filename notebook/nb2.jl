using JLD2, Random, LinearAlgebra, Plots, Statistics

include("../src/init.jl")
include("../config/yaml_loader.jl")
include("../grad/gradient.jl")
include("../src/tools.jl")
config, _ = load_yaml_iter();

raw"""
 1- Init model
 2- Load data
 3- Train
 4- Compute LL

 Train:
 1- Compute the positive phase using data
 2- Compute the negative phase using MALA or argmax
 3- Update parameters
 4- Save weights and biases every epoch
"""

if config.rkm["gpu_usage"]
    try
        CUDA.device!(config.rkm["dev"])
    catch
        @warn "CUDA.device! prompt error. Skipping selecting device"
    end
    # ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]=config.rkm["maxmem"]
end


rkm, J, hparams = initModel(; nv=config.rkm["nv"], 
        nh=config.rkm["nh"], batch_size=config.rkm["batch_size"], 
        lr=config.rkm["lr"], γ=config.rkm["gamma"], t=config.rkm["t"], 
        gpu_usage = config.rkm["gpu_usage"], optType=config.rkm["optType"])
rkm, J, hparams = initModel(; nv=config.rkm["nv"], 
        nh=config.rkm["nh"], batch_size=config.rkm["batch_size"], 
        lr=config.rkm["lr"], γ=config.rkm["gamma"], t=config.rkm["t"], 
        gpu_usage = false, optType=config.rkm["optType"])

x_data,y_data = load_data(config, hparams)

if config.rkm["optType"]=="Adam"
    opt = initOptW(hparams, J) 
elseif optType=="SGD"
    opt = 0
end

dev = selectDev(hparams)

for epoch in 1:1
    for (i,x) in enumerate(x_data)
        # x = Array{Float64}(x) |> dev
        x = CuArray{Float64}(x)
        @info epoch, i, size(x), typeof(x)
        # Δw = compute_gradients_approx(x,rkm,J, config)
        # Δw = compute_gradients_argmax(x,rkm,J, config)
        # Δw = compute_gradients_exact(x,rkm,J, config)
        Δw = compute_gradients(x,rkm,J, config)
        @info mean(Δw)
        updateJAdam!(J, Δw, opt; hparams)
        if i>5
            break
        end
    end
end

saveModel(rkm,J,hparams,config,1)

rkm, J, hparams = loadModel(config, idx=100 )
CuArray{Float64}(rand(10,10)) |> dev
hparams.gpu_usage

@time pos_phase(rkm.v, J, config.rkm["batch_size"]);
@time neg_phase_approx(rkm.v, J, 1, config.rkm["batch_size"]);
@time neg_phase_exact(rkm.v, J, 1, config.rkm["batch_size"]);
@time neg_phase_argmax(rkm.v, J, 1, config.rkm["batch_size"]);
@time neg_phase(rkm.v, J, 1, config.rkm["batch_size"]);

x = CuArray{Float64}(x); #_data[1] |> dev;
@time compute_gradients_approx(x,rkm,J, config);
@time compute_gradients_argmax(x,rkm,J, config);
@time compute_gradients_exact(x,rkm,J, config);
@time compute_gradients(x,rkm,J, config);

28*31*100/60/60

(4*60+30)*31*100/60/60/24

β=1.0
θ = rkm.v
v_to_h(θ,J) .% Float32(2π)
θ = rkm.h
h_to_v(θ,J)

rkm.v = gpu(x_data[1][:,1:400])

θ = CuArray{Float32}(rand(784,400) .* 2π)
θ = rand(784,400) .* 2π
rkm.v
v = cpu(bgs_argmax(rkm.v,J,config.rkm["t"]))
v = cpu(bgs_approx(rkm.v,J,config.rkm["t"]))
v = cpu(bgs(θ,J,50))
bgs(rkm.v,J,1)
v = cpu(v)

lnum=5
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
heatmap(cos.(mat_rot))
heatmap(mat_rot)


rkm.v = gpu(v)
plot(cpu(magnetization(rkm.v))[1,:])

plot(cpu(energy(rkm,J)))

heatmap(reshape(x_data[1][:,1], 28,28))

plot(reshape(cpu(J.w),:), st=:histogram, fillalpha=0.5, normalize=true, bins=100)
##########
using BenchmarkTools
N = 500_000
# generate random batch of 8×3 matrices
A_batch = CUDA.rand(Float32, 8, 3, N)
# compute all pseudoinverses on the GPU
@time P_batch = batched_pinv(A_batch);


s=(1,1_000_000)
@time α, sGL, gammaC, sGU, mT, A, p = CuArray(rand(prod(s))), CuArray(zeros(prod(s))),
        CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64,prod(s),8)),
        CuArray(ones(Float64, 8,3,prod(s))), CuArray(rand(3,prod(s)));
@time α, sGL, gammaC, sGU, mT, A, p = CUDA.rand(prod(s)), CUDA.zeros(prod(s)),
        CUDA.zeros(prod(s)), CUDA.zeros(prod(s)), CUDA.ones(prod(s),8),
        CUDA.ones(8,3,prod(s)), CUDA.rand(3,prod(s))

# @time psi = TikhonovGenGPU(α, sGL, gammaC, sGU, mT, A);
@time psi = TikhonovGenGPUv2(α, sGL, gammaC, sGU, mT, A);

@time p = TikhonovGenGPUv2_A(α, sGL, gammaC, sGU, mT, A);
@time psi = TikhonovGenGPUv2_B(α, sGL, gammaC, sGU, p);

α, sGL, gammaC, sGU, mT, A = Array(ones(prod(s))), Array(zeros(prod(s))),
        Array(zeros(prod(s))),Array(zeros(prod(s))), Array(ones(prod(s),8)),
        Array(ones(Float32, 8,3,prod(s)))
@time psi = TikhonovGen(α, sGL, gammaC, sGU, mT, A);

plot(cpu(psi), st=:histogram)

