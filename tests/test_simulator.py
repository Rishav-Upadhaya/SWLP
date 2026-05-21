from swlp.benchmark.simulator import Scenario, simulate_scenario


def test_simulation_outputs_basic_metrics():
    scenario = Scenario(
        name="unit-test",
        num_layers=4,
        layer_compute_ms=2.0,
        layer_weight_mb=10.0,
        window_sizes=[1, 2],
        context_tokens=64,
        generate_tokens=8,
        kv_bytes_per_token=1024,
        vram_capacity_mb=1024,
        ram_capacity_mb=2048,
        pcie_bandwidth_gbps=16.0,
        ram_bandwidth_gbps=50.0,
        disk_bandwidth_gbps=0.0,
        disk_staging=False,
        overlap=True,
    )

    results = simulate_scenario(scenario)
    assert len(results) == 2
    assert all(result.per_token_seconds > 0 for result in results)
    assert all(result.throughput_tokens_per_second > 0 for result in results)
    assert all(result.recommendation for result in results)


def test_memory_overflow_marks_not_viable():
    scenario = Scenario(
        name="overflow",
        num_layers=2,
        layer_compute_ms=1.0,
        layer_weight_mb=600.0,
        window_sizes=[1],
        context_tokens=128,
        generate_tokens=16,
        kv_bytes_per_token=1024,
        vram_capacity_mb=256,
        ram_capacity_mb=2048,
        pcie_bandwidth_gbps=16.0,
        ram_bandwidth_gbps=50.0,
        disk_bandwidth_gbps=0.0,
        disk_staging=False,
        overlap=True,
    )

    result = simulate_scenario(scenario)[0]
    assert result.fits_vram is False
    assert "not viable" in result.recommendation
