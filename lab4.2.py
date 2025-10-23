import numpy as np
import matplotlib.pyplot as plt

# 1) Згенеруємо синтетичні дані (24 години)
hours = np.arange(0, 24)

# Імітуємо зміну швидкості вітру протягом доби (м/с)
# Більший вітер вдень, менший вночі
wind_speed = 5 + 3 * np.sin(np.pi * (hours - 4) / 12) + np.random.normal(0, 0.5, len(hours))
wind_speed = np.clip(wind_speed, 0, None)  # від’ємний вітер неможливий

# 2) Параметри турбіни
air_density = 1.225        # густина повітря (кг/м³)
rotor_diameter = 40        # діаметр ротора, м
area = np.pi * (rotor_diameter / 2) ** 2  # площа кола, м²
cp = 0.4                   # коефіцієнт потужності (ефективність)
eff_generator = 0.95       # ККД генератора
rated_power_kw = 500       # номінальна потужність турбіни, кВт
cut_in = 3                 # мін. швидкість запуску (м/с)
cut_out = 25               # макс. швидкість (м/с)

# 3) Розрахунок потужності (спрощена кубічна модель)
power_kw = 0.5 * air_density * area * cp * (wind_speed ** 3) / 1000 * eff_generator

# Обмеження за cut-in / cut-out
power_kw[wind_speed < cut_in] = 0
power_kw[wind_speed > cut_out] = 0
power_kw = np.clip(power_kw, 0, rated_power_kw)

# 4) Побудова графіка
plt.figure(figsize=(8, 4))
plt.plot(hours, wind_speed, "o--", label="Швидкість вітру (м/с)")
plt.ylabel("Швидкість вітру, м/с", color="tab:blue")
plt.xlabel("Година доби")
plt.twinx()
plt.plot(hours, power_kw, "s-", color="tab:red", label="Потужність (кВт)")
plt.ylabel("Потужність, кВт", color="tab:red")
plt.title("Добовий профіль вітрової турбіни (синтетика)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Додатково: оцінка добового виробітку (енергії)
energy_kwh = np.sum(power_kw)  # сума за години ~ кВт·год
print(f"Сумарна добова генерація: {energy_kwh:.1f} кВт·год")
