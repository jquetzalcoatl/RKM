using Parameters: @with_kw, @unpack
using CUDA, Flux, JLD2
using MLDatasets

@with_kw struct HyperParams
    nv::Int = 32*32
    nh::Int = 100
    batch_size::Int=100
    lr::Float64 = 0.0001
    γ::Float64 = 0.007
    t::Int = 500
    gpu_usage::Bool = false
    optType::String = "SGD"
end

mutable struct RKM
    v
    h
end

mutable struct Weights
    w
    a
    b
end

mutable struct WeightOpt
    w
    a
    b
end

raw"""
This is taken from Adamopt.jl and slightly adapted.
https://gist.github.com/vankesteren/96207abcd16ecd01a2491bcbec12c73f
"""

mutable struct Adam
  theta # Parameter array
#   loss::Function                # Loss function
#   grad                          # Gradient function
  m #::AbstractArray{Float64}     # First moment
  v #::AbstractArray{Float64}     # Second moment
  b1 #::Float64                   # Exp. decay first moment
  b2 #::Float64                   # Exp. decay second moment
  a #::Float64                    # Step size
  eps #::Float64                  # Epsilon for stability
  t #::Int                        # Time step (iteration)
end

# Outer constructor
# function Adam(theta::AbstractArray{Float64}, loss::Function, grad::Function)
function Adam(theta, a; dev)
  m   = zeros(size(theta)) |> dev
  v   = zeros(size(theta)) |> dev
  b1  = 0.9
  b2  = 0.999
#   a   = 0.001
  eps = 1e-6
  t   = 0
  Adam(theta, m, v, b1, b2, a, eps, t)
end

# Step function with optional keyword arguments for the data passed to grad()
function step!(opt::Adam, Δ)
  opt.t += 1
  gt    = Δ # opt.grad(opt.theta; data...)  #<---- Delta W
  opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* gt
  opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* gt .^ 2
  mhat = opt.m ./ (1 - opt.b1^opt.t)
  vhat = opt.v ./ (1 - opt.b2^opt.t)
  opt.theta = opt.theta + opt.a .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
  opt.theta
end

function selectDev(args)
    if args.gpu_usage
        dev = gpu
    else
        dev = cpu
    end
    dev
end

function genRKM(args)
    # dev = selectDev(args)
    if args.gpu_usage
        return RKM(CUDA.rand(Float64,args.nv, args.batch_size)*2pi, CUDA.rand(Float64,args.nh, args.batch_size)*2pi)
    else
        return RKM(rand(Float64,args.nv, args.batch_size)*2pi, rand(Float64,args.nh, args.batch_size)*2pi)
    end
end

function initWeights(args)
    # dev = selectDev(args)
    if args.gpu_usage
        W = CUDA.randn(Float64,args.nv, args.nh) .* 0.01
        a = CUDA.randn(Float64,args.nv) .* 0.01
        b = CUDA.randn(Float64,args.nh) .* 0.01
    else
        W = randn(Float64,args.nv, args.nh) .* 0.01
        a = randn(Float64,args.nv) .* 0.01
        b = randn(Float64,args.nh) .* 0.01
    end
    return Weights(W,a,b)
end

function initModel(; nv=32*32, nh=100, batch_size=100, lr=0.001, γ=0.001, t=500, gpu_usage = false, optType="SGD")
    hparams = HyperParams(nv, nh, batch_size, lr, γ, t, gpu_usage, optType)
    rkm = genRKM(hparams)
    J = initWeights(hparams)
    return rkm, J, hparams
end

function initOptW(args, J)
    dev = selectDev(args)
    optW = Adam(J.w, args.lr; dev)
    optA = Adam(J.a, args.lr; dev)
    optB = Adam(J.b, args.lr; dev)
    opt = WeightOpt(optW, optA, optB)
    return opt
end


function load_data(config, hparams::HyperParams)
    if config.rkm["dataset"] == "XY"
        x = load(config.rkm["data_path"])["spin_array"]
        x = reshape(x, size(x,1),size(x,2)*size(x,3))
        y = Array(reshape(repeat(reshape(collect(0:9),:,1),1,2000)',:,1)')

        idx = randperm(size(x,2))
        train_data_x = x[:,idx]
        train_data_y = y[:,idx]
        x = [train_data_x[:,i] for i in Iterators.partition(1:size(train_data_x,2), hparams.batch_size)][1:end-1]
        y = [train_data_y[:,i] for i in Iterators.partition(1:size(train_data_y,2), hparams.batch_size)][1:end-1]
        return x,y
    elseif config.rkm["dataset"] == "MNIST"
        train_x = MLDatasets.MNIST(split=:train)[:].features
        train_y = MLDatasets.MNIST(split=:train)[:].targets

        train_x_samp = Array{Float32}(train_x[:, :, train_y .== 0])
        # if size(numbers,1)>1
            # for idx in numbers[2:end]
        train_x_tmp = Array{Float32}(train_x[:, :, train_y .== 1])
        train_x_samp = cat(train_x_samp, train_x_tmp, dims=3)
            # end
        # end
        train_x = train_x_samp * config.rkm["scale"]
        @info size(train_x,3)
        idx = randperm(size(train_x,3))
        train_data = reshape(train_x, 28*28, :)[:,idx]
        x = [train_data[:,i] for i in Iterators.partition(1:size(train_data,2), hparams.batch_size)][1:end-1]
        return x,0
    end 
end

function load_data_test(config, hparams::HyperParams)
    if config.rkm["dataset"] == "XY"
        x = load(config.rkm["data_path"])["spin_array"]
        x = reshape(x, size(x,1),size(x,2)*size(x,3))
        y = Array(reshape(repeat(reshape(collect(0:9),:,1),1,2000)',:,1)')

        idx = randperm(size(x,2))
        train_data_x = x[:,idx]
        train_data_y = y[:,idx]
        x = [train_data_x[:,i] for i in Iterators.partition(1:size(train_data_x,2), hparams.batch_size)][1:end-1]
        y = [train_data_y[:,i] for i in Iterators.partition(1:size(train_data_y,2), hparams.batch_size)][1:end-1]
        return x,y
    elseif config.rkm["dataset"] == "MNIST"
        train_x = MLDatasets.MNIST(split=:test)[:].features
        train_y = MLDatasets.MNIST(split=:test)[:].targets

        train_x_samp = Array{Float32}(train_x[:, :, train_y .== 0])
        # if size(numbers,1)>1
            # for idx in numbers[2:end]
        train_x_tmp = Array{Float32}(train_x[:, :, train_y .== 1])
        train_x_samp = cat(train_x_samp, train_x_tmp, dims=3)
            # end
        # end
        train_x = train_x_samp * config.rkm["scale"]
        @info size(train_x,3)
        idx = randperm(size(train_x,3))
        train_data = reshape(train_x, 28*28, :)[:,idx]
        # x = [train_data[:,i] for i in Iterators.partition(1:size(train_data,2), 2_000)][1:end-1]
        return train_data,0
    end 
end

function saveModel(rkm, J, hparams, config, epoch=0)
    baseDir = config.rkm["model_path"] * "/" * config.rkm["model_name"]
    isdir("/$(baseDir)/J") || mkpath("/$(baseDir)/J")
    # @info "$(baseDir)/models/$path"
    save("$(baseDir)/RKM.jld", "rkm", RKM(Array{Float64}(rkm.v) , Array{Float64}(rkm.h)) )
    save("$(baseDir)/J/J_$(epoch).jld", "J", Weights(Array{Float64}(J.w) , Array{Float64}(J.a), Array{Float64}(J.b)) )
    save("$(baseDir)/hparams.jld", "hparams", hparams)
end

function loadModel(config; idx=-1)
    baseDir = config.rkm["model_path"] * "/" * config.rkm["model_name"]
    isdir(baseDir) || @warn "Dir does not exist"
    @info baseDir
    rkm = load("$(baseDir)/RKM.jld", "rkm")
    if config.rkm["gpu_usage"]
        rkm = RKM([CuArray{Float64}(getfield(rkm, field)) for field in fieldnames(RKM)]...)
    else
        rkm = RKM([Array{Float64}(getfield(rkm, field)) for field in fieldnames(RKM)]...)
    end
    try
        if idx == -1
            JFiles = readdir("$(baseDir)/J/")
            # idx = split(vcat(split.(sort(JFiles), "_")...)[end],".")[1]
            idx = sort(parse.(Int, hcat(split.(hcat(split.(JFiles, "_")...)[2,:], ".")...)[1,:]))[end]
            J = load("$(baseDir)/J/J_$(idx).jld", "J")
        else
            J = load("$(baseDir)/J/J_$(idx).jld", "J")
        end
        @info "Loading model J_$(idx)."
    catch
        # J = load("$(baseDir)/models/$path/J.jld", "J")
        @warn "Instance model does not exist..."
    end
    if config.rkm["gpu_usage"]
        J = Weights([CuArray{Float32}(getfield(J, field)) for field in fieldnames(Weights)]...)
    else
        J = Weights([getfield(J, field) for field in fieldnames(Weights)]...)
    end
    hparams = load("$(baseDir)/hparams.jld", "hparams")
    return rkm, J, hparams
end

# function saveDict(dict; path = "0", baseDir = "/home/javier/Projects/RBM/Results")
#     isdir(baseDir * "/models/$path") || mkpath(baseDir * "/models/$path")
#     save("$(baseDir)/models/$path/dict.jld", "dict", dict)
# end

# function loadDict(path = "0", baseDir = "/home/javier/Projects/RBM/Results")
#     dict = load("$(baseDir)/models/$path/dict.jld", "dict")
#     return dict
# end