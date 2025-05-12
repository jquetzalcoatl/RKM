using Bessels, LinearAlgebra, OMEinsum, CUDA

function Tikhonov_(alpha)
    # ------------------------------------------
    # NOTE: This generates zero-mean variates.
    # alpha: shape parameter
    # S : number of samples desired
    # ------------------------------------------
    S=1
    # Tuning component processes
    ngL = 1
    nc  = 4
    ngU = 7
    nVec = collect(1:7)

    sGL = sqrt(-2 * log(besseli(ngL, alpha) / besseli(0, alpha))) / ngL
    gammaC = -(2 / nc) * log(besseli(nc, alpha) / besseli(0, alpha))
    sGU = sqrt(-2 * log(besseli(ngU, alpha) / besseli(0, alpha))) / ngU

    # Moments of component processes
    mT = besseli.(nVec, alpha) / besseli(0, alpha)
    mGL = exp.(-0.5 * (nVec .* sGL).^2)
    mC = exp.(-0.5 * nVec * gammaC)
    mGU = exp.(-0.5 * (nVec .* sGU).^2)

    # Computing mixture probabilities
    A = hcat([1; mGL], [1; mC], [1; mGU])
    p = pinv(A) * [1; mT]
    p = p / sum(p)

    # Computing logic mixture flags and indexes
    r = rand(S)
    s1 = r .<= p[1]
    s2 = (r .> p[1]) .& (r .<= p[1] + p[2])
    s3 = r .> (p[1] + p[2])

    index1 = findall(s1)
    index2 = findall(s2)
    index3 = findall(s3)

    # Generating component phasors
    xL = sGL * randn(length(index1))
    xC = -(gammaC / 2) * (randn(length(index2)) ./ randn(length(index2)))
    xU = sGU * randn(length(index3))

    rT = zeros(ComplexF64, S)
    rT[index1] .= exp.(im .* xL)
    rT[index2] .= exp.(im .* xC)
    rT[index3] .= exp.(im .* xU)

    # Output: Approximate Tikhonov angular process
    psi = angle.(rT)

    return psi, rT, p
end

function TikhonovGen(alpha, sGL, gammaC, sGU, mT, A)
    tikhonov_loop(alpha, sGL, gammaC, sGU, mT, A)
    A_inv = zeros(size(A,2),size(A,1),size(A,3))
    
    @inbounds for i in 1:size(A,3)
        A_inv[:,:,i] = pinv(A[:,:,i])
    end
    # A_inv = cat([reshape(pinv(A[i,:,:]),1,3,:) for i in 1:length(alpha)]...,dims=1)
    
    @ein p[j,i] := A_inv[j,k,i] * mT[i,k]
    p = p ./ sum(p, dims=1)

    # r = rand(size(alpha,1))
    r = rand(size(alpha,1))
    s1 = r .<= p[1,:]
    s2 = (r .> p[1,:]) .& (r .<= p[1,:] + p[2,:])
    s3 = r .> (p[1,:] + p[2,:])

    index1 = findall(s1)
    index2 = findall(s2)
    index3 = findall(s3)

    # Generating component phasors
    xL = sGL[index1] .* randn(length(index1))
    xC = -(gammaC[index2] / 2) .* (randn(length(index2)) ./ randn(length(index2)))
    xU = sGU[index3] .* randn(length(index3))
    
    rT = zeros(ComplexF64, prod(size(alpha)))
    rT[index1] .= exp.(im .* xL)
    rT[index2] .= exp.(im .* xC)
    rT[index3] .= exp.(im .* xU)

    # Output: Approximate Tikhonov angular process
    # psi = 
    angle.(rT)
end

function TikhonovGenGPU(alpha, sGL, gammaC, sGU, mT, A)
    @cuda tikhonov_loop(alpha, sGL, gammaC, sGU, mT, A)
    A = Array(A)
    A_inv = zeros(size(A,2),size(A,1),size(A,3))
    # A_inv = CuArray(zeros(size(A,1),size(A,3),size(A,2)))
    
    @inbounds for i in 1:size(A,3)
        A_inv[:,:,i] = pinv(A[:,:,i])
    end
    A_inv = CuArray(A_inv)
    @ein p[j,i] := A_inv[j,k,i] * mT[i,k]
    p = p ./ sum(p, dims=1)

    # r = rand(size(alpha,1))
    r = CuArray(rand(size(alpha,1)))
    s1 = r .<= p[1,:]
    s2 = (r .> p[1,:]) .& (r .<= p[1,:] + p[2,:])
    s3 = r .> (p[1,:] + p[2,:])

    index1 = findall(s1)
    index2 = findall(s2)
    index3 = findall(s3)

    # Generating component phasors
    xL = sGL[index1] .* CuArray(randn(length(index1)))
    xC = -(gammaC[index2] / 2) .* CuArray((randn(length(index2)) ./ randn(length(index2))))
    xU = sGU[index3] .* CuArray(randn(length(index3)))
    
    rT = CuArray(zeros(ComplexF64, prod(size(alpha))))
    rT[index1] .= exp.(im .* xL)
    rT[index2] .= exp.(im .* xC)
    rT[index3] .= exp.(im .* xU)

    # Output: Approximate Tikhonov angular process
    angle.(rT)
    # psi
end

function tikhonov_loop(alpha, sGL, gammaC, sGU, mT, A)
    ngL = 1
    nc  = 4
    ngU = 7
    # nVec = collect(1:7)

    @inbounds for i in 1:length(alpha)
        sGL[i] = sqrt(-2 * log(besseli(ngL, alpha[i]) / besseli(0, alpha[i]))) / ngL
        gammaC[i] = -(2 / nc) * log(besseli(nc, alpha[i]) / besseli(0, alpha[i]))
        sGU[i] = sqrt(-2 * log(besseli(ngU, alpha[i]) / besseli(0, alpha[i]))) / ngU

        @inbounds for nVec in 1:7
            # # Moments of component processes
            mT[i,1+nVec] = besseli(nVec, alpha[i]) / besseli(0, alpha[i])
            
            A[1+nVec, 1,i] = exp(-0.5 * (nVec * sGL[i])^2)
            A[1+nVec, 2,i] = exp(-0.5 * nVec * gammaC[i])
            A[1+nVec, 3,i] = exp(-0.5 * (nVec * sGU[i])^2)
        end
    end
end

##############
function TikhonovGenGPUv2(alpha, sGL, gammaC, sGU, mT, A)
    @cuda tikhonov_loop(alpha, sGL, gammaC, sGU, mT, A)
    
    A_inv = batched_pinv(A);
    
    @ein p[j,i] := A_inv[j,k,i] * mT[i,k]
    p = p ./ sum(p, dims=1)

    r = CUDA.rand(Float64, size(alpha,1))

    index1 = CUDA.findall(r .<= p[1,:])
    # @time index1 = CuArray{Int64}(1:length(r))[r .<= p[1,:]]
    # @info size(index1)
    index2 = CUDA.findall((r .> p[1,:]) .& (r .<= p[1,:] + p[1,:]))
    # @info size(index2)
    index3 = CUDA.findall(r .> (p[1,:] + p[2,:]))
    # @info size(index3)

    xL = sGL[index1] .* CUDA.randn(Float64, length(index1))
    xC = -(gammaC[index2] / 2) .* CUDA.randn(Float64, length(index2)) ./ CUDA.randn(Float64, length(index2))
    xU = sGU[index3] .* CUDA.randn(Float64, length(index3))
    
    rT = CuArray(zeros(ComplexF64, prod(size(alpha))))
    rT[index1] .= exp.(im .* xL)
    rT[index2] .= exp.(im .* xC)
    rT[index3] .= exp.(im .* xU)

    angle.(rT)
end

function TikhonovGenGPUv2_A(alpha, sGL, gammaC, sGU, mT, A)
    @cuda tikhonov_loop(alpha, sGL, gammaC, sGU, mT, A)
    
    A_inv = batched_pinv(A);
    
    @ein p[j,i] := A_inv[j,k,i] * mT[i,k]
    p = p ./ sum(p, dims=1)
    return p
end

function TikhonovGenGPUv2_B(alpha, sGL, gammaC, sGU, p)
    @time r = CUDA.rand(Float64, size(alpha,1))

    @time index1 = CUDA.findall(r .<= p[1,:])
    # @time index1 = CuArray{Int64}(1:length(r))[r .<= p[1,:]]
    @info size(index1)
    @time index2 = CUDA.findall((r .> p[1,:]) .& (r .<= p[1,:] + p[1,:]))
    @info size(index2)
    @time index3 = CUDA.findall(r .> (p[1,:] + p[2,:]))
    @info size(index3)

    @time xL = sGL[index1] .* CUDA.randn(Float64, length(index1))
    @time xC = -(gammaC[index2] / 2) .* CUDA.randn(Float64, length(index2)) ./ CUDA.randn(Float64, length(index2))
    @time xU = sGU[index3] .* CUDA.randn(Float64, length(index3))
    
    @time rT = CuArray(zeros(ComplexF64, prod(size(alpha))))
    @time rT[index1] .= exp.(im .* xL)
    @time rT[index2] .= exp.(im .* xC)
    @time rT[index3] .= exp.(im .* xU)

    angle.(rT)
end

# Device‐side helper: compute sqrt(a^2 + b^2) without under/overflow
@inline function pythag(a::Float64, b::Float64)::Float64
    absa = abs(a)
    absb = abs(b)
    if absa > absb
        return absa * sqrt(1f0 + (absb/absa)^2)
    elseif absb == 0f0
        return 0f0
    else
        return absb * sqrt(1f0 + (absa/absb)^2)
    end
end

# GPU kernel: one thread per matrix
function pseudoinv_kernel!(
    A_in::CuDeviceArray{Float64,3},    # (8,3,N)
    A_out::CuDeviceArray{Float64,3}    # (3,8,N)
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = size(A_in, 3)
    if idx <= N
        # 1) form 3×3 Gram = Aᵀ·A
        g11 = 0f0; g12 = 0f0; g13 = 0f0
        g22 = 0f0; g23 = 0f0; g33 = 0f0
        @inbounds for r in 1:8
            a1 = A_in[r,1,idx]
            a2 = A_in[r,2,idx]
            a3 = A_in[r,3,idx]
            g11 += a1*a1; g12 += a1*a2; g13 += a1*a3
            g22 += a2*a2; g23 += a2*a3; g33 += a3*a3
        end

        # 2) invert the symmetric 3×3 via adjugate/determinant
        det = g11*(g22*g33 - g23*g23) -
              g12*(g12*g33 - g23*g13) +
              g13*(g12*g23 - g22*g13)

        inv11 =  (g22*g33 - g23*g23) / det
        inv12 = -(g12*g33 - g23*g13) / det
        inv13 =  (g12*g23 - g22*g13) / det
        inv22 =  (g11*g33 - g13*g13) / det
        inv23 = -(g11*g23 - g12*g13) / det
        inv33 =  (g11*g22 - g12*g12) / det

        # 3) compute pseudo-inverse = (Aᵀ·A)⁻¹ · Aᵀ  →  size 3×8
        @inbounds for r in 1:8
            a1 = A_in[r,1,idx]
            a2 = A_in[r,2,idx]
            a3 = A_in[r,3,idx]
            # row 1
            A_out[1, r, idx] = inv11*a1 + inv12*a2 + inv13*a3
            # row 2
            A_out[2, r, idx] = inv12*a1 + inv22*a2 + inv23*a3
            # row 3
            A_out[3, r, idx] = inv13*a1 + inv23*a2 + inv33*a3
        end
    end
    return
end

"""
    batched_pinv(Ah::Array{Float32,3})

Compute the Moore–Penrose pseudoinverse for a batch of
8×3 matrices stored in `Ah[:,:,i]` for i=1…N,
on the GPU using CUDA.

Returns a 3×8×N Array of pseudoinverses.
"""
function batched_pinv(dA::CuArray{Float64,3})
    # Transfer data to GPU
    # dA = CuArray(Ah)                             # (8,3,N)
    dP = similar(dA, Float64, 3, 8, size(dA,3))  # (3,8,N)

    # launch one thread per matrix
    threads = 1024 #256
    blocks = cld(size(dA,3), threads)
    @cuda threads=threads blocks=blocks pseudoinv_kernel!(dA, dP)

    dP
end