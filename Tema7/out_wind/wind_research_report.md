# Дослідницький проєкт: електрична потужність вітрової турбіни

## 1) Дані та якість

| n_rows | n_turbines | ts_min | ts_max | missing_wind_speed | missing_power |
| --- | --- | --- | --- | --- | --- |
| 19440 | 9 | 2025-01-01 00:00:00 | 2025-03-31 23:00:00 | 0 | 0 |

## 2) Підсумки по турбінах

| site | turbine_id | energy_kwh | power_kw_mean | wind_mps_mean | n | capacity_factor |
| --- | --- | --- | --- | --- | --- | --- |
| Coastal | Coastal-T1 | 2.42867e+06 | 1124.39 | 7.95635 | 2160 | 0.374795 |
| Coastal | Coastal-T2 | 2.40356e+06 | 1112.76 | 7.91755 | 2160 | 0.370919 |
| Coastal | Coastal-T3 | 2.45328e+06 | 1135.78 | 7.96146 | 2160 | 0.378593 |
| Mountain | Mountain-T1 | 1.28759e+06 | 596.106 | 5.83635 | 2160 | 0.198702 |
| Mountain | Mountain-T2 | 1.23367e+06 | 571.146 | 5.70336 | 2160 | 0.190382 |
| Mountain | Mountain-T3 | 1.29043e+06 | 597.422 | 5.8298 | 2160 | 0.199141 |
| Steppe | Steppe-T1 | 1.7774e+06 | 822.87 | 6.76276 | 2160 | 0.27429 |
| Steppe | Steppe-T2 | 1.76018e+06 | 814.899 | 6.72264 | 2160 | 0.271633 |
| Steppe | Steppe-T3 | 1.68461e+06 | 779.914 | 6.65088 | 2160 | 0.259971 |

## 3) Weibull (вітровий ресурс)

| site | turbine_id | weibull_k | weibull_lambda |
| --- | --- | --- | --- |
| Coastal | Coastal-T1 | 2.239 | 8.98663 |
| Coastal | Coastal-T2 | 2.18964 | 8.9413 |
| Coastal | Coastal-T3 | 2.17888 | 8.99066 |
| Mountain | Mountain-T1 | 1.82597 | 6.56831 |
| Mountain | Mountain-T2 | 1.82299 | 6.41513 |
| Mountain | Mountain-T3 | 1.80115 | 6.55918 |
| Steppe | Steppe-T1 | 2.00202 | 7.63147 |
| Steppe | Steppe-T2 | 2.00347 | 7.58968 |
| Steppe | Steppe-T3 | 2.06481 | 7.51075 |

## 4) Performance (індикатор 9–12 м/с)

| site | turbine_id | p_mean_9_12 | perf_index_9_12 |
| --- | --- | --- | --- |
| Coastal | Coastal-T1 | 2017.76 | 0.672587 |
| Coastal | Coastal-T2 | 2044.06 | 0.681353 |
| Coastal | Coastal-T3 | 2033.46 | 0.67782 |
| Mountain | Mountain-T1 | 2011.2 | 0.6704 |
| Mountain | Mountain-T2 | 2017.65 | 0.672551 |
| Mountain | Mountain-T3 | 2027.68 | 0.675893 |
| Steppe | Steppe-T1 | 2022.9 | 0.674298 |
| Steppe | Steppe-T2 | 2028.77 | 0.676257 |
| Steppe | Steppe-T3 | 2012.65 | 0.670884 |

## 5) Економіка (LCOE/NPV/IRR, спрощено)

| site | turbine_id | energy_year1_kwh_est | capex_uah | opex_yearly_uah | price_uah_per_kwh | LCOE_uah_per_kwh | NPV_uah | IRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coastal | Coastal-T1 | 9.84961e+06 | 1.65e+08 | 4.95e+06 | 6 | 2.28723 | 3.46728e+08 | 0.32159 |
| Coastal | Coastal-T2 | 9.74776e+06 | 1.65e+08 | 4.95e+06 | 6 | 2.31113 | 3.40933e+08 | 0.317829 |
| Coastal | Coastal-T3 | 9.94943e+06 | 1.65e+08 | 4.95e+06 | 6 | 2.26428 | 3.52406e+08 | 0.325274 |
| Mountain | Mountain-T1 | 5.22189e+06 | 1.65e+08 | 4.95e+06 | 6 | 4.31421 | 8.34646e+07 | 0.144361 |
| Mountain | Mountain-T2 | 5.00324e+06 | 1.65e+08 | 4.95e+06 | 6 | 4.50276 | 7.10257e+07 | 0.135274 |
| Mountain | Mountain-T3 | 5.23342e+06 | 1.65e+08 | 4.95e+06 | 6 | 4.30471 | 8.41205e+07 | 0.144836 |
| Steppe | Steppe-T1 | 7.20834e+06 | 1.65e+08 | 4.95e+06 | 6 | 3.12531 | 1.96471e+08 | 0.222749 |
| Steppe | Steppe-T2 | 7.13852e+06 | 1.65e+08 | 4.95e+06 | 6 | 3.15588 | 1.92498e+08 | 0.22008 |
| Steppe | Steppe-T3 | 6.83205e+06 | 1.65e+08 | 4.95e+06 | 6 | 3.29745 | 1.75064e+08 | 0.20831 |

## 6) Sensitivity (CAPEX×price) — приклад 10 рядків

| capex_mult | price_mult | NPV_uah | IRR |
| --- | --- | --- | --- |
| 0.7 | 0.7 | 1.2899e+09 | 0.228297 |
| 0.7 | 0.85 | 1.85466e+09 | 0.287512 |
| 0.7 | 1 | 2.41943e+09 | 0.345752 |
| 0.7 | 1.15 | 2.9842e+09 | 0.403529 |
| 0.7 | 1.3 | 3.54896e+09 | 0.461082 |
| 0.85 | 0.7 | 1.00154e+09 | 0.178022 |
| 0.85 | 0.85 | 1.5663e+09 | 0.228297 |
| 0.85 | 1 | 2.13107e+09 | 0.277154 |
| 0.85 | 1.15 | 2.69584e+09 | 0.325268 |
| 0.85 | 1.3 | 3.2606e+09 | 0.37298 |

## 7) Monte-Carlo (summary)

| index | NPV_uah | IRR |
| --- | --- | --- |
| count | 2000 | 2000 |
| mean | 1.86703e+09 | 0.234669 |
| std | 5.44704e+08 | 0.0524358 |
| min | -1.14288e+08 | 0.071612 |
| 5% | 9.78846e+08 | 0.155996 |
| 50% | 1.83949e+09 | 0.229545 |
| 95% | 2.77828e+09 | 0.327565 |
| max | 3.68705e+09 | 0.53178 |


### Графіки

- monthly_energy_by_site.png
- power_curve_example.png
- mc_npv_hist.png
