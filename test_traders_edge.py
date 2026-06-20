"""Offline unit tests for Traders Edge MCP (math + parsing; no network)."""
import math
import numpy as np
import traders_edge_mcp as t


def test_parse_occ():
    r = t._parse_occ("SPXW260619C05500000")
    assert r == ("SPXW", "2026-06-19", "C", 5500.0), r
    r = t._parse_occ("SPX260618P04000000")
    assert r == ("SPX", "2026-06-18", "P", 4000.0), r
    assert t._parse_occ("garbage") is None
    print("ok parse_occ")


def test_year_frac():
    far = t._year_frac("2099-12-31")
    assert far > 70, far
    floor = t._year_frac("2000-01-01")          # past -> floored, still > 0
    assert floor > 0, floor
    print("ok year_frac (far=%.2f, floored=%.2e)" % (far, floor))


def test_norm_cdf_accuracy():
    xs = np.array([-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5])
    approx = t._norm_cdf(xs)
    exact = np.array([0.5 * (1 + math.erf(x / math.sqrt(2))) for x in xs])
    err = float(np.max(np.abs(approx - exact)))
    assert err < 1e-6, err
    assert abs(float(t._norm_cdf(np.array([0.0]))[0]) - 0.5) < 1e-9
    print("ok norm_cdf (max err=%.2e)" % err)


def test_greeks_sanity():
    S = 5000.0
    K = np.array([5000.0, 5000.0, 4800.0, 5200.0])
    T = np.array([5 / 365, 5 / 365, 5 / 365, 5 / 365])
    iv = np.array([0.15, 0.15, 0.15, 0.15])
    isc = np.array([True, False, True, True])
    d, g, vn, cm = t._greeks_vec(S, K, T, iv, isc)
    assert g[0] > 0 and np.all(g > 0), g                       # gamma strictly positive
    assert abs(g[0] - g[1]) < 1e-12                            # call/put gamma identical at same K
    assert 0.45 < d[0] < 0.6, d[0]                             # ATM call delta ~0.5
    assert -0.6 < d[1] < -0.4, d[1]                            # ATM put delta ~-0.5
    assert d[2] > d[0] > d[3]                                  # ITM call > ATM > OTM call delta
    print("ok greeks (atm gamma=%.3e, call=%.3f put=%.3f)" % (g[0], d[0], d[1]))


def test_gex_sign_convention():
    # all calls -> positive GEX; all puts -> negative GEX
    spot = 5000.0
    calls = [{"strike": 5000.0, "cp": "C", "oi": 1000, "iv": 0.15, "expiry": "2099-01-15"}]
    puts = [{"strike": 5000.0, "cp": "P", "oi": 1000, "iv": 0.15, "expiry": "2099-01-15"}]
    cc = t._gex_components(spot, calls)
    pp = t._gex_components(spot, puts)
    assert cc["total"] > 0, cc["total"]
    assert pp["total"] < 0, pp["total"]
    assert abs(cc["total"] + pp["total"]) < 1e-6               # symmetric magnitude
    print("ok gex sign (call=+%.0f, put=%.0f)" % (cc["total"], pp["total"]))


def test_max_pain_and_em():
    opts = [
        {"strike": 4900.0, "cp": "C", "oi": 100, "mid": 110.0},
        {"strike": 5000.0, "cp": "C", "oi": 500, "mid": 40.0},
        {"strike": 5100.0, "cp": "C", "oi": 100, "mid": 8.0},
        {"strike": 4900.0, "cp": "P", "oi": 100, "mid": 9.0},
        {"strike": 5000.0, "cp": "P", "oi": 500, "mid": 38.0},
        {"strike": 5100.0, "cp": "P", "oi": 100, "mid": 105.0},
    ]
    mp = t._max_pain(opts)
    assert mp == 5000.0, mp                                    # heaviest OI strike = pin
    em = t._expected_move(5000.0, opts)
    assert em["atmStrike"] == 5000.0 and em["straddle"] == 78.0, em
    print("ok max_pain=%.0f, expected_move=%.1f pts" % (mp, em["expectedMovePts"]))


if __name__ == "__main__":
    test_parse_occ()
    test_year_frac()
    test_norm_cdf_accuracy()
    test_greeks_sanity()
    test_gex_sign_convention()
    test_max_pain_and_em()
    print("\nALL OFFLINE TESTS PASSED")
