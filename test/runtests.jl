using BlindPolicies
using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using Test

let
    T = 50
    N = 50

    pomdp = BabyPOMDP()

    rsum = 0.0
    tol = 1e-2
    policy = solve(BlindPolicySolver(verbose=true), pomdp)
    @test isapprox(policy.alphas[1][1], (discount(pomdp)*pomdp.p_become_hungry*pomdp.r_hungry/(1-discount(pomdp)))/(1-discount(pomdp)*(1-pomdp.p_become_hungry)), atol=tol) # full without feeding
    @test isapprox(policy.alphas[1][2], pomdp.r_hungry/(1-discount(pomdp)), atol=tol) # hungry without feeding
    @test isapprox(policy.alphas[2][1], pomdp.r_feed/(1-discount(pomdp)), atol=tol) # full with feeding
    @test isapprox(policy.alphas[2][2], pomdp.r_hungry + pomdp.r_feed/(1-discount(pomdp)), atol=tol) # hungry with feeding
    for i in 1:N
        sim = RolloutSimulator(max_steps=T, rng=MersenneTwister(i))
        rsum += simulate(sim, pomdp, policy)
    end

    @show rsum/N
end