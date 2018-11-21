using DynamicalSystemsBase
using DynamicalSystemsBase: CDS

struct MinimalTsit5 end
using StaticArrays
export MinimalTsit5, MinimalTsit5Integrator

mutable struct MinimalTsit5Integrator{IIP, T, S <: AbstractVector{T}, P, F}
  f::F                  # eom
  uprev::S              # previous state
  u::S                  # current state
  tprev::T              # previous time
  t::T                  # current time
  cs::SVector{6, T}     # ci factors cache
  as::SVector{21, T}    # aij factors cache
  tmp::S                # dummy
  p::P                  # parameter container
  t0::T                 # initial time, only for reinit
  dt::T                 # step size
  ks::Vector{S}         # interpolants of the algorithm
end

const MT5I = MinimalTsit5Integrator

function DynamicalSystemsBase.integrator(ds::CDS{IIP, S, D, F, P}, alg::MinimalTsit5;
    u0 = ds.u0, t0 = ds.t0, dt = 0.01
    ) where {IIP, S, D, F, P}

    cs, as = _build_caches(alg, eltype(S))
    ks = [similar(u0) for i in 1:6]

    u = copy(u0)

    integ = MinimalTsit5Integrator{IIP, eltype(S), S, P, F}(
        ds.f, u, u, t0, t0, cs, as, u, ds.p, t0, dt, ks
    )
end

function _build_caches(::MinimalTsit5, ::Type{T}) where {T}

    cs = SVector{6, T}(0.161 ,0.327 ,0.9 ,0.9800255409045097, 1.0, 1.0)

    as = SVector{21, T}(
    convert(T,0.161),
    convert(T,-0.008480655492356989),
    convert(T,0.335480655492357),
    convert(T,2.8971530571054935),
    convert(T,-6.359448489975075),
    convert(T,4.3622954328695815),
    convert(T,5.325864828439257),
    convert(T,-11.748883564062828),
    convert(T,7.4955393428898365),
    convert(T,-0.09249506636175525),
    convert(T,5.86145544294642),
    convert(T,-12.92096931784711),
    convert(T,8.159367898576159),
    convert(T,-0.071584973281401),
    convert(T,-0.028269050394068383),
    convert(T,0.09646076681806523),
    convert(T,0.01),
    convert(T,0.4798896504144996),
    convert(T,1.379008574103742),
    convert(T,-3.290069515436081),
    convert(T,2.324710524099774))

    #btilde1 = convert(T,-0.00178001105222577714)
    #btilde2 = convert(T,-0.0008164344596567469)
    #btilde3 = convert(T,0.007880878010261995)
    #btilde4 = convert(T,-0.1447110071732629)
    #btilde5 = convert(T,0.5823571654525552)
    #btilde6 = convert(T,-0.45808210592918697)
    #btilde7 = convert(T,0.015151515151515152)

    # see here for interpolation
    # https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/blob/master/src/dense/interpolants.jl#L254-L270
    # evalpoly is "something like"  ((((2+x)x+5)x+6)x+2)x+1

    return cs, as
end

function step!(integ::MinimalTsit5Integrator{true, T, S}) where {T, S}

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as

    k1, k2, k3, k4, k5, k6 = integ.ks; k7 = k1
    tmp = integ.tmp; f! = integ.f

    integ.uprev .= integ.u; uprev = integ.u

    @. tmp = uprev+dt*a21*k1
    f!(k2, tmp, p, t+c1*dt)
    @. tmp = uprev+dt*(a31*k1+a32*k2)
    f!(k3, tmp, p, t+c2*dt)
    @. tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
    f!(k4, tmp, p, t+c3*dt)
    @. tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
    f!(k5, tmp, p, t+c4*dt)
    @. tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
    f!(k6, tmp, p, t+dt)

    @. integ.u = uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)

    f!(k7, integ.u, p, t+dt)

    integ.t += dt

    return  nothing
end

# %%
ds = Systems.lorenz_iip()
integ = integrator(ds, MinimalTsit5(); dt = 0.01)
step!(integ)

@profiler for i in 1:1000000; step!(integ); end
#
# using PyPlot
# N = 100000
# xs = zeros(N); ys = copy(xs); zs = copy(xs)
#
# for i in 1:N
#     step!(integ)
#     xs[i], ys[i], zs[i] = integ.u
# end
#
# plot3D(xs, ys, zs)
using BenchmarkTools
function bench()
    ds = Systems.lorenz_iip()
    integ = integrator(ds, MinimalTsit5(); dt = 0.01)
    step!(integ)
    @btime step!($integ)
end

bench()
