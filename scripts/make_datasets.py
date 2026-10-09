"""Generate the bundled sample datasets in data/ (synthetic, reproducible, license-free).

    python scripts/make_datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# --- Regression: house prices -------------------------------------------------
n = 600
area = rng.normal(1600, 450, n).clip(450, 4200).round()
bedrooms = np.clip((area / 550 + rng.normal(0, 0.8, n)).round(), 1, 6).astype(int)
bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0, 0.5, n)).round(), 1, 4).astype(int)
age = rng.integers(0, 80, n)
distance = rng.gamma(2.2, 4.0, n).round(1)
neighborhood = rng.choice(["north", "south", "east", "west", "downtown"], n, p=[.22, .2, .2, .2, .18])
garage = rng.choice(["yes", "no"], n, p=[.65, .35])
premium = pd.Series(neighborhood).map({"north": 25, "south": -10, "east": 0, "west": 10, "downtown": 60}).to_numpy()
price = (55 + area * 0.16 + bedrooms * 6 + bathrooms * 9 - age * 0.45 - distance * 2.1
         + premium + (garage == "yes") * 14 + rng.normal(0, 18, n)).round(1)  # thousands of $
houses = pd.DataFrame({
    "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age,
    "distance_to_center_km": distance, "neighborhood": neighborhood, "garage": garage,
    "price_k": price,
})
houses.loc[rng.choice(n, 12, replace=False), "age_years"] = np.nan  # a few gaps on purpose
houses.to_csv(DATA / "house-prices.csv", index=False)

# --- Classification: customer churn -------------------------------------------
n = 800
tenure = rng.integers(1, 72, n)
contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[.55, .25, .2])
monthly = rng.normal(70, 22, n).clip(18, 120).round(2)
internet = rng.choice(["fiber", "dsl", "none"], n, p=[.45, .4, .15])
support_calls = rng.poisson(1.6, n)
paperless = rng.choice(["yes", "no"], n, p=[.6, .4])
logit = (-1.0 - tenure * 0.05 + (contract == "month-to-month") * 1.6 - (contract == "two-year") * 1.2
         + (monthly - 70) * 0.02 + support_calls * 0.45 + (internet == "fiber") * 0.5 + rng.normal(0, .6, n))
churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
churn = pd.DataFrame({
    "tenure_months": tenure, "contract": contract, "monthly_charges": monthly,
    "total_charges": (monthly * tenure).round(2), "internet_service": internet,
    "support_calls": support_calls, "paperless_billing": paperless, "churned": churned,
})
churn.to_csv(DATA / "customer-churn.csv", index=False)
print(houses.shape, churn.shape, churn.churned.value_counts().to_dict())
