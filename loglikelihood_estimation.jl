using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures, JLD2
using CUDA

include("utils/train.jl")
include("scripts/RAIS.jl")
include("configs/yaml_loader.jl")
config, _ = load_yaml_iter();

if config.gpu["gpu_bool"]
    dev = gpu
    CUDA.device_reset!()
    CUDA.device!(config.gpu["gpu_id"])
else
    dev = cpu
end

function estimate_LL(PATH::String, v_test::Matrix{Float32}, config)
    
    s = size(readdir("$(PATH)/J"),1)
    l=config.analysis["num_models"]
    ais = Vector{Float64}(undef, 0)
    rais = Vector{Float64}(undef, 0)
    num = Vector{Float64}(undef, 0)
    LL_R = Vector{Float64}(undef, 0)
    LL_A = Vector{Float64}(undef, 0)

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        
        rkm, J, hparams = loadModel(config, idx=idx )
        
        push!(ais, AIS(rkm,J,hparams, config.analysis["nbeta"]))
        push!(rais, RAIS(rkm,J,hparams, config.analysis["nbeta"]) )
        push!(num, LL_numerator(v_test,J))
        push!(LL_R, num[end] - rais[end])
        push!(LL_A, num[end] - ais[end])
    end

    return ais, rais, num, LL_R, LL_A
end

function plot_and_save(PATH::String, ais, rais, num, LL_R, LL_A)

    isdir("$(PATH)/Figs") || mkpath("$(PATH)/Figs")

    f = plot( ais, xscale=:identity, color=:blue, label="AIS", markershape=:circle)
    f = plot!( rais, color=:black, label="reverse AIS", s=:auto, markershapes = :square, lw=0, markerstrokewidth=0)
    f = plot!(size=(700,500), xlabel="Epochs", frame=:box, ylabel="log(Z)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/ais_and_rais_$(modelname).png")
    
    f = plot( LL_A, xscale=:identity, color=:blue, label="loglikelihood AIS", markershape=:circle)
    f = plot!( LL_R, xscale=:identity, color=:magenta, label="loglikelihood RAIS", markershape=:square)
    f = plot!(size=(700,500), xlabel="Epochs ", frame=:box, ylabel="LL", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/loglikelihood_ais_rais_$(modelname).png")
    
    jldsave("$(PATH)/Figs/partition_analytics.jld", rais=rais, ais=ais, num=num, llr=LL_R, lla=LL_A)
    
end

function main()
    if config.rkm["gpu_usage"]
        try
            CUDA.device!(config.rkm["dev"])
        catch
            @warn "CUDA.device! prompt error. Skipping selecting device"
        end
        ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]=config.rkm["maxmem"]
    end
    
    @info "Initializing model..."
    rkm, J, hparams = initModel(; nv=config.rkm["nv"], 
            nh=config.rkm["nh"], batch_size=config.rkm["batch_size"], 
            lr=config.rkm["lr"], γ=config.rkm["gamma"], t=config.rkm["t"], 
            gpu_usage = config.rkm["gpu_usage"], optType=config.rkm["optType"])
    @info "Loading data..."
    x_data,y_data = load_data_test(config, hparams)

    PATH = config.rkm["model_path"] * "/" * config.rkm["model_name"]
    modelname = config.rkm["model_name"]
    
    @info modelname
    ais, rais, num, LL_R, LL_A = estimate_LL(PATH, x_data, config);
    plot_and_save(PATH, ais, rais, num, LL_R, LL_A)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end