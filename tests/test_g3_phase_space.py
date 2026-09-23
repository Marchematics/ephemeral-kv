from benchmarks.g3_phase_space import crossover_tokens, phase_grid


def test_crossover_moves_later_on_faster_links():
    slow = crossover_tokens(
        kv_bytes_per_token=32 * 1024,
        link_bandwidth_gbs=25,
        prefill_tokens_per_s=4000,
        active_tokens=4096,
        lookup_ms=10,
    )
    fast = crossover_tokens(
        kv_bytes_per_token=32 * 1024,
        link_bandwidth_gbs=50,
        prefill_tokens_per_s=4000,
        active_tokens=4096,
        lookup_ms=10,
    )
    assert fast == 2 * slow


def test_crossover_moves_earlier_for_larger_kv_per_token():
    small = crossover_tokens(
        kv_bytes_per_token=32 * 1024,
        link_bandwidth_gbs=25,
        prefill_tokens_per_s=4000,
        active_tokens=4096,
        lookup_ms=10,
    )
    large = crossover_tokens(
        kv_bytes_per_token=128 * 1024,
        link_bandwidth_gbs=25,
        prefill_tokens_per_s=4000,
        active_tokens=4096,
        lookup_ms=10,
    )
    assert large < small
    assert small // large == 4


def test_smaller_active_sets_expand_the_mobility_region():
    rows = phase_grid(
        kv_kib=(32,), link_gbs=(25,), active_tokens=(2048, 4096, 8192)
    )
    by_active = {r["active_tokens"]: r["crossover_history_tokens"] for r in rows}
    assert by_active[2048] < by_active[4096] < by_active[8192]
