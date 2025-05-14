include("./src/logger.jl")
# include("../rkm/init.jl")
# include("../config/yaml_loader.jl")
# include("../rkm/gradient.jl")
# include("../rkm/tools.jl")
include("./src/init.jl")
include("./config/yaml_loader.jl")
include("./grad/gradient.jl")
include("./src/tools.jl")
config, _ = load_yaml_iter();

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
    # rkm, J, hparams = initModel(; nv=config.rkm["nv"], 
    #         nh=config.rkm["nh"], batch_size=config.rkm["batch_size"], 
    #         lr=config.rkm["lr"], γ=config.rkm["gamma"], t=config.rkm["t"], 
    #         gpu_usage = false, optType=config.rkm["optType"])
    @info "Loading data..."
    x_data,y_data = load_data(config, hparams)
    
    if config.rkm["optType"]=="Adam"
        opt = initOptW(hparams, J) 
    elseif optType=="SGD"
        opt = 0
    end
    
    dev = selectDev(hparams)
    @info "Training starts..."
    if hparams.gpu_usage
        for epoch in 1:config.rkm["epochs"]
            @info epoch
            for (i,x) in enumerate(x_data)
                x = CuArray{Float64}(x)
                @info epoch, i, size(x), typeof(x)
                # Δw = compute_gradients_approx(x,rkm,J, config)
                # Δw = compute_gradients_argmax(x,rkm,J, config)
                # Δw = compute_gradients_exact(x,rkm,J, config)
                Δw = compute_gradients(x,rkm,J, config)
                @info mean(Δw)
                updateJAdam!(J, Δw, opt; hparams)
            end
            saveModel(rkm,J,hparams,config,epoch)
        end 
    else
        for epoch in 1:config.rkm["epochs"]
            @info epoch
            for (i,x) in enumerate(x_data)
                x = Array{Float64}(x)
                @info epoch, i, size(x), typeof(x)
                # Δw = compute_gradients_approx(x,rkm,J, config)
                # Δw = compute_gradients_argmax(x,rkm,J, config)
                # Δw = compute_gradients_exact(x,rkm,J, config)
                Δw = compute_gradients(x,rkm,J, config)
                @info mean(Δw)
                updateJAdam!(J, Δw, opt; hparams)
            end
            saveModel(rkm,J,hparams,config,epoch)
        end 
    end
    @info "Training ends" 
end

if abspath(PROGRAM_FILE) == @__FILE__
    # julia train.jl
    @info "Number of threads $(Threads.nthreads())"
    main()
end