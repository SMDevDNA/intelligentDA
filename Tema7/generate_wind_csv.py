import numpy as np
import pandas as pd

def power_curve_kw(v, rated_kw=3000.0, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)

    m1 = (v >= cut_in) & (v < rated_speed)
    p[m1] = rated_kw * ((v[m1]**3 - cut_in**3) / (rated_speed**3 - cut_in**3))

    m2 = (v >= rated_speed) & (v < cut_out)
    p[m2] = rated_kw

    return p

def weibull(rng, k, lam, n):
    u = rng.random(n)
    return lam * (-np.log(1 - u)) ** (1 / k)

def generate(out_csv="wind_fact_hourly.csv", start="2025-01-01 00:00:00", days=90, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")

    sites = {
        "Coastal":  (2.2, 9.0),
        "Steppe":   (2.0, 7.5),
        "Mountain": (1.8, 6.5),
    }
    turbines = ["T1", "T2", "T3"]

    rows = []
    for site, (k, lam) in sites.items():
        for t in turbines:
            v = weibull(rng, k, lam, len(idx))
            p_kw = power_curve_kw(v)

            df = pd.DataFrame({
                "timestamp": idx,
                "site": site,
                "turbine_id": f"{site}-{t}",
                "wind_speed_mps": v,
                "power_electric_kw": p_kw,
                "energy_kwh": p_kw  # 1 година -> kWh = kW * 1h
            })
            rows.append(df)

    data = pd.concat(rows, ignore_index=True)
    data.to_csv(out_csv, index=False)
    print(f"OK: saved {out_csv} rows={len(data)} cols={len(data.columns)}")

if __name__ == "__main__":
    generate()
