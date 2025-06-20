using JLD2, Random, LinearAlgebra, Plots, Statistics

include("../src/init.jl")
include("../config/yaml_loader.jl")
include("../grad/gradient.jl")
include("../src/tools.jl")
include("../src/RAIS.jl")
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



x_data,y_data = load_data_test(config, hparams)

x_data

AIS(rkm,J,hparams,30.0)
RAIS(rkm,J,hparams,30.0)

PATH = config.rkm["model_path"] * "/" * config.rkm["model_name"]

rkm, J, hparams = loadModel(config, idx=500 )

β=1.0
θ = CuArray{Float64}(x_data[1][:,1:2000])
a,b = A_h(θ,J,β), B_h(θ,J,β)
k_h=CUDA.sqrt.( a .^ 2 .+ b .^ 2)
besseli.(0,k_h)

mean(sum(log.(besseli.(0,k_h)),dims=1))
log(2π)

MLDatasets.MNIST(split=:test)[:].features