"""The fixed e-commerce schema and a deterministic builder for its SQLite database.

The database is generated from a seeded RNG, so every machine (your laptop, Colab,
Kaggle) builds a byte-for-byte identical `db/shop.db`. That matters: execution
accuracy compares query *results*, so the data must be the same everywhere.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db" / "shop.db"
SEED = 1812

SCHEMA_SQL = """\
CREATE TABLE categories (
  category_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  parent_category_id INTEGER REFERENCES categories(category_id) -- NULL for top-level categories
);

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  category_id INTEGER NOT NULL REFERENCES categories(category_id),
  brand TEXT NOT NULL,
  price REAL NOT NULL,          -- current list price in USD
  cost REAL NOT NULL,           -- unit cost to the store in USD
  stock_quantity INTEGER NOT NULL,
  is_active INTEGER NOT NULL,   -- 1 = listed, 0 = discontinued
  created_at TEXT NOT NULL      -- 'YYYY-MM-DD'
);

CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  city TEXT NOT NULL,
  country TEXT NOT NULL,
  loyalty_tier TEXT NOT NULL,   -- 'bronze', 'silver', 'gold', 'platinum'
  signup_date TEXT NOT NULL     -- 'YYYY-MM-DD'
);

CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  order_date TEXT NOT NULL,     -- 'YYYY-MM-DD'
  status TEXT NOT NULL,         -- 'pending', 'shipped', 'delivered', 'cancelled', 'returned'
  shipping_cost REAL NOT NULL
);

CREATE TABLE order_items (
  order_item_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(order_id),
  product_id INTEGER NOT NULL REFERENCES products(product_id),
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL,     -- price actually charged per unit
  discount REAL NOT NULL        -- fraction 0.0-0.3 taken off unit_price
);

CREATE TABLE payments (
  payment_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(order_id),
  payment_date TEXT NOT NULL,   -- 'YYYY-MM-DD'
  method TEXT NOT NULL,         -- 'credit_card', 'paypal', 'debit_card', 'gift_card', 'bank_transfer'
  amount REAL NOT NULL,
  status TEXT NOT NULL          -- 'completed', 'refunded', 'failed'
);

CREATE TABLE reviews (
  review_id INTEGER PRIMARY KEY,
  product_id INTEGER NOT NULL REFERENCES products(product_id),
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  rating INTEGER NOT NULL,      -- 1 to 5
  review_date TEXT NOT NULL     -- 'YYYY-MM-DD'
);
"""

# Notes appended to the schema in every prompt (training and eval alike).
SCHEMA_NOTES = """\
-- Notes:
-- Revenue of an order line = quantity * unit_price * (1 - discount).
-- Dates are ISO strings; use strftime('%Y', ...) / strftime('%Y-%m', ...) for grouping.
-- Data covers 2023-01-01 to 2025-12-31."""


def schema_prompt() -> str:
    """The schema text shown to every model."""
    return SCHEMA_SQL.strip() + "\n\n" + SCHEMA_NOTES


_CATEGORIES = {
    "Electronics": ["Laptops", "Smartphones", "Headphones", "Cameras"],
    "Home & Kitchen": ["Cookware", "Furniture", "Bedding"],
    "Sports & Outdoors": ["Fitness", "Camping", "Cycling"],
    "Books": ["Fiction", "Non-Fiction", "Children's Books"],
    "Beauty": ["Skincare", "Haircare"],
}
_BRANDS = {
    "Laptops": ["Nimbus", "Voltix", "Arcbook"], "Smartphones": ["Voltix", "Pebble", "Orbit"],
    "Headphones": ["Sonora", "Pebble", "Hushwave"], "Cameras": ["Lumen", "Aperture Co"],
    "Cookware": ["IronLeaf", "ChefLine"], "Furniture": ["Oakhaus", "Nordform"],
    "Bedding": ["CloudRest", "Nordform"], "Fitness": ["PulseFit", "IronLeaf"],
    "Camping": ["TrailKind", "Summit Gear"], "Cycling": ["Velocity", "Summit Gear"],
    "Fiction": ["Lantern Press", "Blue Owl"], "Non-Fiction": ["Lantern Press", "Meridian"],
    "Children's Books": ["Blue Owl", "Kitebook"], "Skincare": ["Dewdrop", "Botanica"],
    "Haircare": ["Botanica", "Silkline"],
}
_NOUNS = {
    "Laptops": ["Ultrabook 13", "Pro 15", "Air 14", "Gaming 17"],
    "Smartphones": ["X1", "Lite", "Max", "Mini"],
    "Headphones": ["Studio Wireless", "Buds", "Over-Ear ANC", "Sport Buds"],
    "Cameras": ["Mirrorless M5", "Action Cam", "Instant Snap"],
    "Cookware": ["Cast Iron Skillet", "Nonstick Pan Set", "Dutch Oven"],
    "Furniture": ["Desk Chair", "Standing Desk", "Bookshelf", "Coffee Table"],
    "Bedding": ["Duvet", "Pillow Pair", "Sheet Set"],
    "Fitness": ["Yoga Mat", "Adjustable Dumbbells", "Resistance Bands", "Kettlebell"],
    "Camping": ["2-Person Tent", "Sleeping Bag", "Camp Stove", "Headlamp"],
    "Cycling": ["Road Helmet", "Bike Lights", "Saddle Bag"],
    "Fiction": ["The Quiet Harbor", "Glass Orchard", "Night Train North"],
    "Non-Fiction": ["Deep Work Habits", "A Short History of Maps", "Data for Everyone"],
    "Children's Books": ["The Brave Little Fox", "Counting Stars", "Moon Picnic"],
    "Skincare": ["Hydrating Serum", "SPF 50 Sunscreen", "Night Cream"],
    "Haircare": ["Repair Shampoo", "Leave-In Conditioner", "Hair Oil"],
}
_PRICE_RANGE = {
    "Electronics": (25, 1800), "Home & Kitchen": (15, 600), "Sports & Outdoors": (10, 350),
    "Books": (8, 40), "Beauty": (6, 70),
}
_FIRST = ["Aarav", "Maya", "Liam", "Sofia", "Noah", "Priya", "Ethan", "Chloe", "Lucas", "Aisha",
          "Mateo", "Emma", "Kenji", "Olivia", "Omar", "Isabella", "Yuki", "Zara", "Diego", "Hannah",
          "Ravi", "Grace", "Leo", "Fatima", "Jonas", "Mia", "Arjun", "Elena", "Samuel", "Nora"]
_LAST = ["Sharma", "Smith", "Garcia", "Kim", "Mueller", "Rossi", "Patel", "Nguyen", "Brown",
         "Silva", "Tanaka", "Johnson", "Khan", "Lopez", "Novak", "Dubois", "Ali", "Wilson",
         "Costa", "Ivanova"]
_CITIES = {
    "USA": ["New York", "Austin", "Seattle", "Chicago", "Denver"],
    "India": ["Mumbai", "Pune", "Bengaluru", "Delhi"],
    "UK": ["London", "Manchester", "Edinburgh"],
    "Germany": ["Berlin", "Munich", "Hamburg"],
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro"],
    "Japan": ["Tokyo", "Osaka"],
    "Australia": ["Sydney", "Melbourne"],
}
_COUNTRY_WEIGHTS = [30, 18, 12, 10, 10, 7, 7, 6]

START, END = date(2023, 1, 1), date(2025, 12, 31)


def _rand_date(rng: random.Random, lo: date = START, hi: date = END) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def build_db(path: Path = DB_PATH, force: bool = False) -> Path:
    """Create the SQLite database deterministically. No-op if it already exists."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(SEED)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)

    # categories: top-level first, then children
    cat_rows, sub_to_top, cat_id = [], {}, 1
    top_ids = {}
    for top in _CATEGORIES:
        cat_rows.append((cat_id, top, None))
        top_ids[top] = cat_id
        cat_id += 1
    sub_ids = {}
    for top, subs in _CATEGORIES.items():
        for sub in subs:
            cat_rows.append((cat_id, sub, top_ids[top]))
            sub_ids[sub] = cat_id
            sub_to_top[sub] = top
            cat_id += 1
    con.executemany("INSERT INTO categories VALUES (?,?,?)", cat_rows)

    # products
    products, pid = [], 1
    for sub, nouns in _NOUNS.items():
        lo, hi = _PRICE_RANGE[sub_to_top[sub]]
        for noun in nouns:
            for brand in _BRANDS[sub]:
                price = round(rng.uniform(lo, hi), 2)
                cost = round(price * rng.uniform(0.35, 0.75), 2)
                products.append((
                    pid, f"{brand} {noun}", sub_ids[sub], brand, price, cost,
                    rng.choice([0, 0, rng.randint(1, 20)]) if rng.random() < 0.12 else rng.randint(20, 500),
                    0 if rng.random() < 0.08 else 1,
                    _rand_date(rng, date(2022, 6, 1), date(2025, 6, 30)).isoformat(),
                ))
                pid += 1
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", products)

    # customers
    customers, emails = [], set()
    countries = list(_CITIES)
    for cid in range(1, 601):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        email = f"{first}.{last}{cid}@example.com".lower()
        emails.add(email)
        country = rng.choices(countries, weights=_COUNTRY_WEIGHTS)[0]
        tier = rng.choices(["bronze", "silver", "gold", "platinum"], weights=[50, 28, 15, 7])[0]
        customers.append((cid, first, last, email, rng.choice(_CITIES[country]), country, tier,
                          _rand_date(rng, date(2022, 1, 1), date(2025, 10, 31)).isoformat()))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)", customers)

    # orders, items, payments
    orders, items, payments = [], [], []
    oid = iid = payid = 1
    active_products = [p for p in products]
    for cust in customers:
        signup = date.fromisoformat(cust[7])
        lo = max(signup, START)
        n_orders = rng.choices(range(0, 13), weights=[8, 14, 14, 12, 10, 9, 8, 7, 6, 4, 3, 3, 2])[0]
        for _ in range(n_orders):
            od = _rand_date(rng, lo, END)
            status = rng.choices(["delivered", "shipped", "pending", "cancelled", "returned"],
                                 weights=[62, 12, 6, 12, 8])[0]
            ship = rng.choice([0.0, 0.0, 4.99, 7.99, 12.5])
            orders.append((oid, cust[0], od.isoformat(), status, ship))
            total = ship
            for prod in rng.sample(active_products, rng.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]):
                qty = rng.choices([1, 2, 3, 4], weights=[70, 18, 8, 4])[0]
                unit = round(prod[4] * rng.uniform(0.9, 1.05), 2)
                disc = rng.choice([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
                items.append((iid, oid, prod[0], qty, unit, disc))
                total += qty * unit * (1 - disc)
                iid += 1
            if status != "pending":
                method = rng.choices(["credit_card", "paypal", "debit_card", "gift_card", "bank_transfer"],
                                     weights=[45, 22, 18, 7, 8])[0]
                pstatus = "refunded" if status in ("cancelled", "returned") else (
                    "failed" if rng.random() < 0.03 else "completed")
                pdate = od + timedelta(days=rng.randint(0, 2))
                payments.append((payid, oid, pdate.isoformat(), method, round(total, 2), pstatus))
                payid += 1
            oid += 1
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
    con.executemany("INSERT INTO payments VALUES (?,?,?,?,?,?)", payments)

    # reviews: only for delivered purchases, so the data stays internally consistent
    delivered = {o[0]: o for o in orders if o[3] == "delivered"}
    reviews, rid, seen = [], 1, set()
    for it in items:
        o = delivered.get(it[1])
        if o is None or rng.random() > 0.35 or (it[2], o[1]) in seen:
            continue
        seen.add((it[2], o[1]))
        rd = min(date.fromisoformat(o[2]) + timedelta(days=rng.randint(3, 40)), END)
        reviews.append((rid, it[2], o[1], rng.choices([1, 2, 3, 4, 5], weights=[6, 8, 16, 34, 36])[0],
                        rd.isoformat()))
        rid += 1
    con.executemany("INSERT INTO reviews VALUES (?,?,?,?,?)", reviews)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the deterministic shop.db")
    ap.add_argument("--force", action="store_true", help="rebuild even if it exists")
    args = ap.parse_args()
    p = build_db(force=args.force)
    con = sqlite3.connect(p)
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        print(f"{t:12s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>6d} rows")
    print(f"-> {p}")
