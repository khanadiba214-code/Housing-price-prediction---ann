import numpy as np
import pandas as pd

def generate_housing_data(output_path="housing_data.csv", n=2400, random_state=42):
    rng = np.random.default_rng(random_state)
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
              "Pune", "Kolkata", "Ahmedabad", "Chandigarh"]
    city_rate = {
        "Mumbai": 22000, "Delhi": 12000, "Bengaluru": 10500,
        "Chennai": 9000, "Hyderabad": 8500, "Pune": 8000,
        "Kolkata": 6500, "Ahmedabad": 5500, "Chandigarh": 7500
    }
    city = rng.choice(cities, n, p=[.13,.13,.14,.10,.12,.12,.09,.08,.09])
    locality_tier = rng.choice(["Premium", "Mid", "Budget"], n, p=[.22,.50,.28])
    property_type = rng.choice(["Apartment", "Independent House", "Villa"], n, p=[.68,.22,.10])
    area_sqft = np.clip(rng.lognormal(mean=7.05, sigma=.48, size=n), 350, 7500).round().astype(int)
    bedrooms = np.clip(np.rint(area_sqft / 550 + rng.normal(0, .7, n)), 1, 8).astype(int)
    bathrooms = np.clip(bedrooms + rng.choice([-1,0,0,1], n), 1, 7).astype(int)
    balconies = rng.integers(0, 4, n)
    age_years = rng.integers(0, 36, n)
    total_floors = rng.integers(1, 31, n)
    floor = np.array([rng.integers(0, f + 1) for f in total_floors])
    furnishing = rng.choice(["Unfurnished", "Semi-Furnished", "Furnished"], n)
    parking = rng.integers(0, 4, n)
    facing = rng.choice(["North", "South", "East", "West"], n)
    near_metro = rng.choice(["Yes", "No"], n, p=[.48,.52])
    lift = rng.choice(["Yes", "No"], n, p=[.78,.22])

    tier_multiplier = np.select(
        [locality_tier == "Premium", locality_tier == "Mid"],
        [1.55, 1.0], default=.72
    )
    type_multiplier = np.select(
        [property_type == "Villa", property_type == "Independent House"],
        [1.35, 1.12], default=1.0
    )
    furnishing_bonus = np.select(
        [furnishing == "Furnished", furnishing == "Semi-Furnished"],
        [.12, .06], default=0
    )
    price = (
        area_sqft * np.array([city_rate[c] for c in city]) *
        tier_multiplier * type_multiplier *
        (1 + furnishing_bonus) *
        (1 + (near_metro == "Yes") * .07) *
        (1 - age_years * .003) *
        rng.normal(1, .16, n)
    )
    price = np.maximum(price, 500000).round(0)

    df = pd.DataFrame({
        "city": city, "locality_tier": locality_tier,
        "property_type": property_type, "area_sqft": area_sqft,
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "balconies": balconies, "age_years": age_years,
        "floor": floor, "total_floors": total_floors,
        "furnishing": furnishing, "parking": parking,
        "facing": facing, "near_metro": near_metro,
        "lift": lift, "price": price
    })

    # Add realistic missing values in selected input columns.
    for col, fraction in {
        "parking": .08, "facing": .07, "floor": .05,
        "age_years": .05, "balconies": .04, "furnishing": .035
    }.items():
        idx = rng.choice(df.index, int(n * fraction), replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} | Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_housing_data()
