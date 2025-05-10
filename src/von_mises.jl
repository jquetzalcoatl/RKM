
ρ(τ,k) = (τ-(2*τ)^0.5)/(2*k)
τ(k)=1+(1+4*k^2)^0.5

function vonMises_sample(μ,k)
    ##Best DJ, Fisher NI. Efficient simulation of the von Mises distribution. 
    ##Journal of the Royal Statistical Society: Series C (Applied Statistics). 
    ##1979 Jun;28(2):152-7.
    u1,u2,u3=rand(3)
    r = (1+ρ(τ(k),k)^2)/(2*ρ(τ(k),k))
    z = cos(π*u1)
    f=(1+r*z)/(r+z)
    c=k*(r-f)
    if c*(2-c)-u2 > 0
        return sign(u3-0.5)*acos(f) + μ
    else
        if log(c/u2)+1-c<0
            vonMises_sample(μ,k)
        else
            return sign(u3-0.5)*acos(f) + μ
        end
    end
end