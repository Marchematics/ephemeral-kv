from benchmarks.g3_hardware_primitives import kv_payload_bytes, summarize_times


def test_kv_payload_bytes_scales_linearly():
    assert kv_payload_bytes(1_000, 32 * 1024) == 32_768_000
    assert kv_payload_bytes(2_000, 32 * 1024) == 65_536_000


def test_time_summary_is_stable():
    s = summarize_times([1.0, 2.0, 3.0])
    assert s["n"] == 3
    assert s["p50_s"] == 2.0
    assert s["min_s"] == 1.0
