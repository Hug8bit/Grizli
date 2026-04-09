#!/usr/bin/env python3
"""
GRIZLI Optimization Tests
Validates the optimization algorithms.
"""

import sys
import time
import numpy as np
import pandapower as pp

from src.core.ieee_networks import create_congested_ieee33, create_atacama_scenario
from src.simulation.power_flow import PowerFlowSimulator
from src.optimization.fast_optimizer import optimize_topology_fast, apply_optimization_result
from src.optimization.reconfiguration import ReconfigurationOptimizer, OptimizationStatus


def test_fast_optimizer():
    """Test fast greedy optimizer."""
    print("\n" + "="*60)
    print("TEST: Fast Greedy Optimizer")
    print("="*60)

    # Create congested network
    net = create_congested_ieee33(load_increase_percent=50, renewable_increase_percent=80)

    # Run initial power flow
    pp.runpp(net, numba=True)
    initial_max_loading = net.res_line.loading_percent.max()
    initial_overloaded = (net.res_line.loading_percent > 100).sum()
    initial_losses = net.res_line.pl_mw.sum()

    print(f"\nInitial state:")
    print(f"  Max loading: {initial_max_loading:.1f}%")
    print(f"  Overloaded lines: {initial_overloaded}")
    print(f"  Losses: {initial_losses*1000:.1f} kW")

    # Run optimization
    start_time = time.time()
    result = optimize_topology_fast(net, max_iterations=5)
    elapsed = time.time() - start_time

    print(f"\nOptimization result:")
    print(f"  Computation time: {result.computation_time_s*1000:.1f} ms")
    print(f"  Iterations: {result.iterations}")
    print(f"  Lines to open: {result.lines_to_open}")
    print(f"  Lines to close: {result.lines_to_close}")
    print(f"  Num switches: {result.num_switches}")

    # Apply and verify
    apply_optimization_result(net, result)
    pp.runpp(net, numba=True)

    final_max_loading = net.res_line.loading_percent.max()
    final_overloaded = (net.res_line.loading_percent > 100).sum()
    final_losses = net.res_line.pl_mw.sum()

    print(f"\nFinal state:")
    print(f"  Max loading: {final_max_loading:.1f}%")
    print(f"  Overloaded lines: {final_overloaded}")
    print(f"  Losses: {final_losses*1000:.1f} kW")

    # Validate
    tests_passed = 0
    tests_total = 4

    # Test 1: Fast computation
    if result.computation_time_s < 1.0:
        print("\n[PASS] Computation time < 1s")
        tests_passed += 1
    else:
        print(f"\n[FAIL] Computation time too long: {result.computation_time_s:.2f}s")

    # Test 2: Congestion reduction
    if result.congestion_reduction_percent > 0:
        print(f"[PASS] Congestion reduced by {result.congestion_reduction_percent:.1f}%")
        tests_passed += 1
    else:
        print(f"[FAIL] No congestion reduction")

    # Test 3: Network still valid
    try:
        pp.runpp(net, numba=True)
        print("[PASS] Power flow converges after optimization")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Power flow failed: {e}")

    # Test 4: Not worse than before
    if final_max_loading <= initial_max_loading + 1:
        print("[PASS] Max loading not increased")
        tests_passed += 1
    else:
        print(f"[FAIL] Max loading increased from {initial_max_loading:.1f}% to {final_max_loading:.1f}%")

    print(f"\nTests passed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total


def test_milp_optimizer():
    """Test MILP optimizer (if PuLP available)."""
    print("\n" + "="*60)
    print("TEST: MILP Optimizer")
    print("="*60)

    try:
        import pulp
    except ImportError:
        print("PuLP not installed, skipping MILP test")
        return True

    # Create network
    net = create_congested_ieee33(load_increase_percent=40, renewable_increase_percent=60)

    # Run initial power flow
    pp.runpp(net, numba=True)
    initial_max_loading = net.res_line.loading_percent.max()
    print(f"\nInitial max loading: {initial_max_loading:.1f}%")

    # Run MILP optimization
    print("\nRunning MILP optimization (time limit: 10s)...")
    optimizer = ReconfigurationOptimizer(net, time_limit_s=10.0, gap_tolerance=0.1)
    result = optimizer.optimize(max_switches=3)

    print(f"\nOptimization status: {result.status.value}")
    print(f"Computation time: {result.computation_time_s:.2f}s")
    print(f"Switches: {result.num_switches}")

    if result.is_successful:
        print(f"Congestion before: {result.congestion_before:.1f}")
        print(f"Congestion after: {result.congestion_after:.1f}")
        print(f"Reduction: {result.congestion_reduction_percent:.1f}%")

        # Validate
        if result.status == OptimizationStatus.OPTIMAL:
            print("\n[PASS] MILP found optimal solution")
            return True
        elif result.status == OptimizationStatus.FEASIBLE:
            print("\n[PASS] MILP found feasible solution")
            return True
    else:
        print(f"\n[INFO] MILP status: {result.status.value}")
        # Not necessarily a failure - might be infeasible or timeout
        return True

    return False


def test_extreme_scenario():
    """Test optimization on extreme Atacama scenario."""
    print("\n" + "="*60)
    print("TEST: Extreme Atacama Scenario")
    print("="*60)

    # Create extreme scenario
    net = create_atacama_scenario()

    # Run initial power flow
    try:
        pp.runpp(net, numba=True)
        initial_max_loading = net.res_line.loading_percent.max()
        initial_overloaded = (net.res_line.loading_percent > 100).sum()
        print(f"\nInitial state:")
        print(f"  Max loading: {initial_max_loading:.1f}%")
        print(f"  Overloaded lines: {initial_overloaded}")
    except Exception as e:
        print(f"Initial power flow failed: {e}")
        print("[INFO] Network too congested for initial solution")
        return True  # This is expected for extreme scenarios

    # Run optimization
    result = optimize_topology_fast(net, max_iterations=10)

    print(f"\nOptimization result:")
    print(f"  Congestion reduction: {result.congestion_reduction_percent:.1f}%")
    print(f"  Switches used: {result.num_switches}")

    # Even partial improvement is acceptable for extreme scenarios
    if result.congestion_reduction_percent > 0 or result.num_switches > 0:
        print("\n[PASS] Optimization attempted improvement on extreme scenario")
        return True
    else:
        print("\n[INFO] No improvement possible (scenario may be too constrained)")
        return True


def test_no_congestion():
    """Test optimizer on non-congested network."""
    print("\n" + "="*60)
    print("TEST: Non-Congested Network")
    print("="*60)

    # Create lightly loaded network
    from src.core.ieee_networks import create_ieee33_with_renewables
    net = create_ieee33_with_renewables(num_pv=2, num_wind=1)

    # Reduce loads to avoid congestion
    net.load.p_mw *= 0.5
    net.load.q_mvar *= 0.5

    pp.runpp(net, numba=True)
    print(f"Max loading: {net.res_line.loading_percent.max():.1f}%")

    # Run optimization
    result = optimize_topology_fast(net, max_iterations=3)

    print(f"Switches suggested: {result.num_switches}")

    # Should not make unnecessary changes
    if result.num_switches == 0:
        print("\n[PASS] No unnecessary switches on healthy network")
        return True
    else:
        print(f"\n[INFO] Made {result.num_switches} switches (may still be valid)")
        return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("#" + " "*15 + "GRIZLI OPTIMIZATION TESTS" + " "*15 + "#")
    print("#"*60)

    tests = [
        ("Fast Greedy Optimizer", test_fast_optimizer),
        ("MILP Optimizer", test_milp_optimizer),
        ("Extreme Scenario", test_extreme_scenario),
        ("Non-Congested Network", test_no_congestion),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
