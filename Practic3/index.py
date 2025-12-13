import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Helper from environment (available in python_user_visible) to show DataFrames nicely
def display_dataframe_to_user(name, dataframe):
    print(f"=== {name} ===")
    print(dataframe)

np.random.seed(42)

# ---------------------- Simulate source (Access) tables ----------------------
n_products = 12
n_employees = 8
n_orders = 400

# Products table (includes Brand and Category)
brands = ["Acme", "Zenith", "Orion", "Nova"]
categories = ["Electronics", "Household", "Sport", "Toys"]
products = pd.DataFrame({
    "product_id": range(1, n_products+1),
    "product_name": [f"Product_{i}" for i in range(1, n_products+1)],
    "brand": np.random.choice(brands, n_products),
    "category": np.random.choice(categories, n_products),
})

# Employees table (LastName, City, Region, Country)
cities = ["Kyiv", "Lviv", "Odesa", "Kharkiv", "Dnipro"]
regions = ["Kyivska", "Lvivska", "Odeska", "Kharkivska", "Dnipropetrovska"]
countries = ["Ukraine", "Poland", "Germany"]
employees = pd.DataFrame({
    "employee_id": range(1, n_employees+1),
    "last_name": [f"Surname_{i}" for i in range(1, n_employees+1)],
    "city": np.random.choice(cities, n_employees),
    "region": np.random.choice(regions, n_employees),
    "country": np.random.choice(countries, n_employees),
})

# Orders table with order_date and assigned employee
start_date = datetime(2024, 1, 1)
orders = pd.DataFrame({
    "order_id": range(1, n_orders+1),
    "order_date": [start_date + timedelta(days=int(x)) for x in np.random.exponential(scale=120, size=n_orders).cumsum() % 365],
    "employee_id": np.random.choice(employees["employee_id"], n_orders),
    # Could include customer info if needed
})

# OrderDetails table linking orders and products with quantity and unit price
order_details_rows = []
for oid in orders["order_id"]:
    # each order has 1..4 line items
    for _ in range(np.random.randint(1,5)):
        pid = int(np.random.choice(products["product_id"]))
        qty = int(np.random.poisson(2) + 1)  # mostly small quantities
        unit_price = float(np.round(np.random.uniform(10, 400), 2))
        order_details_rows.append({"order_id": oid, "product_id": pid, "quantity": qty, "unit_price": unit_price})

order_details = pd.DataFrame(order_details_rows)

# ---------------------- Build star schema (fact + dimensions) ----------------------
# Fact: fact_order_lines (one row per order detail) with measures: quantity, unit_price, total_price
fact = order_details.merge(orders, on="order_id", how="left").merge(products, on="product_id", how="left").merge(employees, on="employee_id", how="left")
fact["total_price"] = fact["quantity"] * fact["unit_price"]

# Date dimension: expand date parts and an 'assignment date' perspective (order_date is the 'date of assignment')
date_dim = (
    fact[["order_date"]]
    .drop_duplicates()
    .assign(
        date = lambda df: df["order_date"].dt.date,
        year = lambda df: df["order_date"].dt.year,
        quarter = lambda df: df["order_date"].dt.quarter,
        month = lambda df: df["order_date"].dt.month,
        month_name = lambda df: df["order_date"].dt.strftime("%b"),
        day = lambda df: df["order_date"].dt.day,
        weekday = lambda df: df["order_date"].dt.day_name(),
    )
    .reset_index(drop=True)
)

# Product dimension
product_dim = products.copy()

# Employee dimension
employee_dim = employees.copy()

# Show shapes and a small sample of each table
meta = {
    "products": products.shape,
    "employees": employees.shape,
    "orders": orders.shape,
    "order_details": order_details.shape,
    "fact_order_lines": fact.shape,
    "date_dim": date_dim.shape,
    "product_dim": product_dim.shape,
    "employee_dim": employee_dim.shape,
}

print("Star schema objects and shapes:")
for k, v in meta.items():
    print(f" - {k}: {v}")

# Display main tables to the user (if display helper available)
if display_dataframe_to_user:
    display_dataframe_to_user("Products (dimension)", products.head(50))
    display_dataframe_to_user("Employees (dimension)", employees.head(50))
    display_dataframe_to_user("Orders (source)", orders.head(50))
    display_dataframe_to_user("Order details (source)", order_details.head(50))
    display_dataframe_to_user("Fact - order lines (fact table)", fact.head(200))
else:
    # Fallback printing small previews
    print("\nProducts (dimension) sample:")
    print(products.head())
    print("\nEmployees (dimension) sample:")
    print(employees.head())
    print("\nFact sample:")
    print(fact.head())

# ---------------------- OLAP-like aggregations (pivot tables) ----------------------
# 1) Aggregate price and quantity by order assignment date (year-month) and brand
fact["year"] = fact["order_date"].dt.year
fact["month"] = fact["order_date"].dt.month
fact["year_month"] = fact["order_date"].dt.to_period("M").astype(str)

agg_brand_time = fact.groupby(["year_month", "brand"]).agg(total_quantity=("quantity", "sum"), total_revenue=("total_price", "sum")).reset_index()

# 2) Aggregate by product category and employee (last name)
agg_category_employee = fact.groupby(["category", "last_name"]).agg(total_quantity=("quantity", "sum"), total_revenue=("total_price", "sum")).reset_index()

# 3) Aggregate by employee location (city, region, country) and month
agg_employee_loc_time = fact.groupby(["year_month", "city", "region", "country"]).agg(total_quantity=("quantity", "sum"), total_revenue=("total_price", "sum")).reset_index()

# Display aggregates
if display_dataframe_to_user:
    display_dataframe_to_user("Agg by Year-Month and Brand", agg_brand_time.head(200))
    display_dataframe_to_user("Agg by Category and Employee", agg_category_employee.head(200))
    display_dataframe_to_user("Agg by Employee Location and Month", agg_employee_loc_time.head(200))
else:
    print("\nAgg by Year-Month and Brand sample:")
    print(agg_brand_time.head())
    print("\nAgg by Category and Employee sample:")
    print(agg_category_employee.head())

# ---------------------- Pivot-like views (analogs of OLAP slicing/dicing) ----------------------
# Example pivot: pivot table where rows = year_month, columns = brand, values = total_revenue
pivot_revenue_brand = agg_brand_time.pivot(index="year_month", columns="brand", values="total_revenue").fillna(0)
pivot_quantity_brand = agg_brand_time.pivot(index="year_month", columns="brand", values="total_quantity").fillna(0)

if display_dataframe_to_user:
    display_dataframe_to_user("Pivot - Revenue by YearMonth x Brand", pivot_revenue_brand.reset_index().head(200))
    display_dataframe_to_user("Pivot - Quantity by YearMonth x Brand", pivot_quantity_brand.reset_index().head(200))
else:
    print("\nPivot - Revenue by YearMonth x Brand (sample):")
    print(pivot_revenue_brand.head())

# ---------------------- Charts ----------------------
# Chart 1: Monthly total revenue (line chart)
monthly_revenue = fact.groupby("year_month").agg(month_revenue=("total_price", "sum")).reset_index().sort_values("year_month")
plt.figure(figsize=(10,4))
plt.plot(monthly_revenue["year_month"], monthly_revenue["month_revenue"], marker='o')
plt.title("Monthly Total Revenue (assignment date)")
plt.xlabel("Year-Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 2: Top 8 products by revenue (bar chart)
top_products = fact.groupby(["product_id", "product_name"]).agg(revenue=("total_price", "sum")).reset_index().sort_values("revenue", ascending=False).head(8)
plt.figure(figsize=(10,4))
plt.bar(top_products["product_name"], top_products["revenue"])
plt.title("Top 8 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 3: Revenue by Category and Employee region (stacked bar)
cat_region = fact.groupby(["category", "region"]).agg(revenue=("total_price", "sum")).reset_index()
pivot_cat_region = cat_region.pivot(index="region", columns="category", values="revenue").fillna(0)
plt.figure(figsize=(10,4))
pivot_cat_region.plot(kind="bar", stacked=True, legend=True)
plt.title("Revenue by Region and Category")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save results to CSV for the user to inspect / import into icCube later
out_dir = "/mnt/data/olap_example_outputs"
os.makedirs(out_dir, exist_ok=True)
pivot_revenue_brand.reset_index().to_csv(os.path.join(out_dir, "pivot_revenue_brand.csv"), index=False)
agg_category_employee.to_csv(os.path.join(out_dir, "agg_category_employee.csv"), index=False)
agg_employee_loc_time.to_csv(os.path.join(out_dir, "agg_employee_loc_time.csv"), index=False)

print(f"\nSaved sample output CSVs to: {out_dir}")
print("If you have your Access DB (.accdb or .mdb), you can replace the simulated tables by loading them (see comments in code).")