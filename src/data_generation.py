"""Synthetic data generation for ecommerce pricing analytics.

This module creates the first reproducible raw customer and transaction
datasets for a Gymshark-style ecommerce analytics project. It uses
probability-based behavioral logic and does not build any machine learning
models.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import exp, log1p
from pathlib import Path
from random import Random
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
PRODUCT_EVENTS_FILE = RAW_DATA_DIR / "product_events.csv"

RANDOM_SEED = 42
DEFAULT_CUSTOMER_COUNT = 12_500
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
PRICE_EVENT_START_DATE = date(2025, 1, 1)
DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class SchemaField:
    """Describe one field in the synthetic analytics dataset schema."""

    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class CustomerBehaviorInputs:
    """Inputs used to estimate purchase and churn probabilities."""

    price_increase_occurred: bool
    discount_percent: float
    purchase_frequency: float
    customer_tenure_days: int
    prior_spending: float
    customer_region: str
    acquisition_channel: str


@dataclass(frozen=True)
class BehaviorConfig:
    """Configurable weights for the synthetic behavioral probability model."""

    purchase_intercept: float = -0.35
    churn_intercept: float = -1.60
    price_increase_purchase_effect: float = -0.40
    price_increase_churn_effect: float = 0.45
    discount_purchase_effect: float = 1.10
    discount_churn_effect: float = -0.35
    frequency_purchase_effect: float = 0.65
    frequency_churn_effect: float = -0.70
    tenure_purchase_effect: float = 0.20
    tenure_churn_effect: float = -0.35
    spending_purchase_effect: float = 0.35
    spending_churn_effect: float = -0.45
    min_probability: float = 0.02
    max_probability: float = 0.95


@dataclass(frozen=True)
class ProductCategory:
    """Represent one synthetic apparel category template used to create products."""

    name: str
    base_price: float
    price_stddev: float
    floor_price: float
    weight: float
    price_increase_tendency: float
    price_elasticity: float


@dataclass(frozen=True)
class Product:
    """Represent one clearly synthetic Gymshark-style product/SKU."""

    product_id: str
    product_name: str
    product_category: str
    unit_cost: float
    list_price: float
    price_stddev: float
    floor_price: float
    weight: float
    price_increase_tendency: float
    price_elasticity: float
    starting_inventory: int


@dataclass(frozen=True)
class ProductPriceContext:
    """Represent the product-day price shown before purchase behavior is realized."""

    list_price: float
    selling_price: float
    discount_percent: float
    price_increase_occurred: bool


@dataclass(frozen=True)
class PriceDemandComparison:
    """Summarize demand movement on normal versus price-increase product-days."""

    normal_average_selling_price: float
    increased_average_selling_price: float
    average_price_change: float
    normal_unit_velocity: float
    increased_unit_velocity: float
    unit_velocity_change: float
    normal_revenue_per_day: float
    increased_revenue_per_day: float
    revenue_per_day_change: float


@dataclass(frozen=True)
class GenerationSummary:
    """Summarize the generated raw dataset."""

    customer_rows: int
    transaction_rows: int
    product_rows: int
    product_event_rows: int
    start_date: date
    end_date: date
    average_order_value: float
    average_gross_margin_rate: float
    churn_rate: float
    price_change_frequency: float
    total_product_views: int
    add_to_cart_rate: float
    checkout_start_rate: float
    product_purchase_rate: float
    stockout_event_rate: float
    price_demand_comparison: PriceDemandComparison
    customers_path: Path
    transactions_path: Path
    products_path: Path
    product_events_path: Path


DEFAULT_BEHAVIOR_CONFIG = BehaviorConfig()


# Intended confounders for later causal inference analysis:
# - customer_region and acquisition_channel may influence who sees pricing or
#   discount changes while also affecting baseline purchase and churn behavior.
# - purchase_frequency, customer_tenure_days, and prior_spending capture customer
#   engagement/value; these can affect treatment targeting and independently
#   influence future purchasing or churn.
# - price_increase_occurred is the main treatment indicator, while
#   discount_percent may be modeled as a treatment, mediator, or moderator
#   depending on the later causal question.
REGION_PURCHASE_EFFECTS: dict[str, float] = {
    "uk": 0.10,
    "north_america": 0.05,
    "europe": 0.00,
    "asia_pacific": -0.03,
    "rest_of_world": -0.08,
}

REGION_CHURN_EFFECTS: dict[str, float] = {
    "uk": -0.05,
    "north_america": 0.00,
    "europe": 0.04,
    "asia_pacific": 0.07,
    "rest_of_world": 0.10,
}

ACQUISITION_PURCHASE_EFFECTS: dict[str, float] = {
    "direct": 0.14,
    "email": 0.12,
    "organic_search": 0.08,
    "influencer": 0.06,
    "paid_social": 0.00,
    "affiliate": -0.02,
}

ACQUISITION_CHURN_EFFECTS: dict[str, float] = {
    "direct": -0.16,
    "email": -0.12,
    "organic_search": -0.08,
    "influencer": -0.02,
    "paid_social": 0.08,
    "affiliate": 0.06,
}

REGION_WEIGHTS: dict[str, float] = {
    "uk": 0.34,
    "north_america": 0.30,
    "europe": 0.19,
    "asia_pacific": 0.10,
    "rest_of_world": 0.07,
}

ACQUISITION_CHANNEL_WEIGHTS: dict[str, float] = {
    "paid_social": 0.28,
    "influencer": 0.20,
    "organic_search": 0.18,
    "direct": 0.16,
    "email": 0.10,
    "affiliate": 0.08,
}

REGION_PRICE_EXPOSURE_EFFECTS: dict[str, float] = {
    "uk": 0.08,
    "north_america": 0.12,
    "europe": 0.02,
    "asia_pacific": -0.05,
    "rest_of_world": -0.10,
}

ACQUISITION_PRICE_EXPOSURE_EFFECTS: dict[str, float] = {
    "direct": 0.08,
    "email": 0.12,
    "organic_search": 0.03,
    "influencer": 0.02,
    "paid_social": 0.04,
    "affiliate": -0.02,
}

ACQUISITION_DISCOUNT_EFFECTS: dict[str, float] = {
    "direct": -0.05,
    "email": 0.12,
    "organic_search": -0.02,
    "influencer": 0.02,
    "paid_social": 0.04,
    "affiliate": 0.08,
}

PRODUCT_CATEGORY_TEMPLATES: tuple[ProductCategory, ...] = (
    ProductCategory("leggings", 52.0, 8.0, 24.0, 0.18, 0.08, -1.20),
    ProductCategory("sports_bras", 38.0, 6.0, 18.0, 0.13, 0.04, -1.10),
    ProductCategory("training_tops", 34.0, 7.0, 15.0, 0.17, 0.00, -1.25),
    ProductCategory("hoodies", 66.0, 11.0, 32.0, 0.13, 0.10, -0.85),
    ProductCategory("joggers", 58.0, 9.0, 28.0, 0.12, 0.07, -1.05),
    ProductCategory("shorts", 36.0, 6.0, 16.0, 0.10, -0.02, -1.35),
    ProductCategory("outerwear", 88.0, 14.0, 42.0, 0.07, 0.14, -0.75),
    ProductCategory("accessories", 22.0, 5.0, 8.0, 0.10, -0.06, -1.45),
)

SYNTHETIC_PRODUCT_NAMES: dict[str, tuple[str, ...]] = {
    "leggings": (
        "Apex Sculpt Leggings",
        "Core Flex Leggings",
        "Motion Seamless Leggings",
        "Lift Training Leggings",
        "Contour High-Rise Leggings",
        "Pulse Pocket Leggings",
    ),
    "sports_bras": (
        "Apex Support Sports Bra",
        "Core Strappy Sports Bra",
        "Motion Longline Sports Bra",
        "Lift High-Support Bra",
        "Contour V-Neck Sports Bra",
    ),
    "training_tops": (
        "Apex Training Tee",
        "Core Fitted Tank",
        "Motion Crop Tee",
        "Lift Performance Top",
        "Everyday Oversized Tee",
        "Tempo Long Sleeve Top",
    ),
    "hoodies": (
        "Apex Rest Day Hoodie",
        "Core Pullover Hoodie",
        "Motion Zip Hoodie",
        "Lift Oversized Hoodie",
        "Tempo Fleece Hoodie",
    ),
    "joggers": (
        "Apex Slim Joggers",
        "Core Fleece Joggers",
        "Motion Training Joggers",
        "Lift Tapered Joggers",
        "Tempo Woven Joggers",
    ),
    "shorts": (
        "Apex Training Shorts",
        "Core Running Shorts",
        "Motion Bike Shorts",
        "Lift Hybrid Shorts",
        "Tempo Woven Shorts",
    ),
    "outerwear": (
        "Apex Training Jacket",
        "Core Puffer Vest",
        "Motion Lightweight Jacket",
        "Lift Zip Jacket",
    ),
    "accessories": (
        "Core Crew Socks",
        "Apex Training Cap",
        "Motion Duffel Bag",
        "Lift Wrist Wraps",
        "Tempo Water Bottle",
    ),
}

DISCOUNT_WEIGHTS: dict[float, float] = {
    10.0: 0.36,
    15.0: 0.28,
    20.0: 0.22,
    25.0: 0.09,
    30.0: 0.05,
}

CUSTOMER_FIELDNAMES = (
    "customer_id",
    "first_purchase_date",
    "customer_region",
    "acquisition_channel",
    "customer_tenure_days",
    "purchase_frequency",
    "prior_spending",
    "transaction_count",
    "total_spending",
    "average_discount_percent",
    "price_increase_occurred",
    "churned",
    "last_transaction_date",
)

TRANSACTION_FIELDNAMES = (
    "transaction_id",
    "customer_id",
    "transaction_date",
    "product_id",
    "product_name",
    "product_category",
    "product_price",
    "unit_cost",
    "list_price",
    "selling_price",
    "discount_amount",
    "discount_percent",
    "quantity",
    "order_value",
    "gross_margin",
    "customer_region",
    "acquisition_channel",
    "customer_tenure_days",
    "purchase_frequency",
    "prior_spending",
    "price_increase_occurred",
    "churned",
)

PRODUCT_FIELDNAMES = (
    "product_id",
    "product_name",
    "product_category",
    "unit_cost",
    "list_price",
    "gross_margin",
    "gross_margin_rate",
    "total_revenue",
    "total_units_sold",
    "total_purchases",
    "product_views",
    "add_to_cart_events",
    "checkout_started_events",
    "inventory_level",
    "stockout_flag",
)

PRODUCT_EVENT_FIELDNAMES = (
    "event_date",
    "product_id",
    "product_name",
    "product_category",
    "unit_cost",
    "list_price",
    "selling_price",
    "discount_percent",
    "product_views",
    "add_to_cart_events",
    "checkout_started_events",
    "purchases",
    "units_sold",
    "revenue",
    "gross_margin",
    "inventory_level",
    "stockout_flag",
    "price_increase_occurred",
)

SYNTHETIC_DATASET_SCHEMA: tuple[SchemaField, ...] = (
    SchemaField(
        name="customer_id",
        dtype="string",
        description="Unique identifier for each synthetic customer.",
    ),
    SchemaField(
        name="transaction_id",
        dtype="string",
        description="Unique identifier for each synthetic transaction.",
    ),
    SchemaField(
        name="transaction_date",
        dtype="date",
        description="Date when the synthetic transaction occurred.",
    ),
    SchemaField(
        name="product_id",
        dtype="string",
        description="Unique identifier for each synthetic product/SKU.",
    ),
    SchemaField(
        name="product_name",
        dtype="string",
        description="Clearly synthetic Gymshark-style product name.",
    ),
    SchemaField(
        name="product_category",
        dtype="string",
        description="Gymshark-style apparel or accessories category.",
    ),
    SchemaField(
        name="product_price",
        dtype="float",
        description="Listed unit product price before transaction-level discounts.",
    ),
    SchemaField(
        name="unit_cost",
        dtype="float",
        description="Synthetic unit cost used to calculate gross margin.",
    ),
    SchemaField(
        name="list_price",
        dtype="float",
        description="Synthetic product list price before discount.",
    ),
    SchemaField(
        name="selling_price",
        dtype="float",
        description="Synthetic unit selling price after discount.",
    ),
    SchemaField(
        name="discount_amount",
        dtype="float",
        description="Absolute discount applied to the transaction.",
    ),
    SchemaField(
        name="discount_percent",
        dtype="float",
        description="Discount applied as a percentage of listed product price.",
    ),
    SchemaField(
        name="quantity",
        dtype="integer",
        description="Number of units purchased in the transaction.",
    ),
    SchemaField(
        name="order_value",
        dtype="float",
        description="Final transaction value after quantity and discount effects.",
    ),
    SchemaField(
        name="gross_margin",
        dtype="float",
        description="Revenue after subtracting synthetic product unit cost.",
    ),
    SchemaField(
        name="customer_region",
        dtype="string",
        description="Broad region assigned to the synthetic customer.",
    ),
    SchemaField(
        name="acquisition_channel",
        dtype="string",
        description="Marketing or discovery channel associated with the customer.",
    ),
    SchemaField(
        name="customer_tenure_days",
        dtype="integer",
        description="Number of days since the synthetic customer's first purchase.",
    ),
    SchemaField(
        name="purchase_frequency",
        dtype="float",
        description="Annualized repeat purchase activity before the observation.",
    ),
    SchemaField(
        name="prior_spending",
        dtype="float",
        description="Cumulative customer spending before the current observation.",
    ),
    SchemaField(
        name="price_increase_occurred",
        dtype="boolean",
        description="Indicator for whether a price increase is present for analysis.",
    ),
    SchemaField(
        name="churned",
        dtype="boolean",
        description="Indicator for whether the synthetic customer churned.",
    ),
    SchemaField(
        name="product_views",
        dtype="integer",
        description="Synthetic product detail page views for funnel analysis.",
    ),
    SchemaField(
        name="add_to_cart_events",
        dtype="integer",
        description="Synthetic product add-to-cart events.",
    ),
    SchemaField(
        name="checkout_started_events",
        dtype="integer",
        description="Synthetic checkout-start events attributed to each product.",
    ),
    SchemaField(
        name="purchases",
        dtype="integer",
        description="Synthetic product-level purchase events.",
    ),
    SchemaField(
        name="inventory_level",
        dtype="integer",
        description="Synthetic product inventory level after daily sales and restocks.",
    ),
    SchemaField(
        name="stockout_flag",
        dtype="boolean",
        description="Indicator for product-day stockout conditions.",
    ),
)


def _sigmoid(score: float) -> float:
    """Convert a linear score into a probability."""

    return 1 / (1 + exp(-score))


def _clip_probability(
    probability: float,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> float:
    """Keep generated probabilities away from deterministic zero or one values."""

    return min(max(probability, config.min_probability), config.max_probability)


def _normalize_discount(discount_percent: float) -> float:
    """Normalize discount input whether expressed as 20 or 0.20."""

    discount_rate = discount_percent / 100 if discount_percent > 1 else discount_percent
    return min(max(discount_rate, 0.0), 0.80)


def _normalize_purchase_frequency(purchase_frequency: float) -> float:
    """Normalize repeat purchase activity for use in probability scoring."""

    return min(max(purchase_frequency, 0.0), 6.0) / 6.0


def _normalize_tenure(customer_tenure_days: int) -> float:
    """Normalize tenure using diminishing returns for longer customer histories."""

    return min(log1p(max(customer_tenure_days, 0)) / log1p(730), 1.0)


def _normalize_prior_spending(prior_spending: float) -> float:
    """Normalize prior spending using diminishing returns for high-value customers."""

    return min(log1p(max(prior_spending, 0.0)) / log1p(2_000), 1.0)


def _weighted_choice(options: dict[str, float], rng: Random) -> str:
    """Select one string option using explicit weights."""

    threshold = rng.random() * sum(options.values())
    cumulative_weight = 0.0
    for option, weight in options.items():
        cumulative_weight += weight
        if threshold <= cumulative_weight:
            return option

    return next(reversed(options))


def _weighted_float_choice(options: dict[float, float], rng: Random) -> float:
    """Select one numeric option using explicit weights."""

    threshold = rng.random() * sum(options.values())
    cumulative_weight = 0.0
    for option, weight in options.items():
        cumulative_weight += weight
        if threshold <= cumulative_weight:
            return option

    return next(reversed(options))


def _choose_product(catalog: tuple[Product, ...], rng: Random) -> Product:
    """Select a synthetic product according to its demand weight."""

    threshold = rng.random() * sum(product.weight for product in catalog)
    cumulative_weight = 0.0
    for product in catalog:
        cumulative_weight += product.weight
        if threshold <= cumulative_weight:
            return product

    return catalog[-1]


def _random_date(rng: Random, start_date: date, end_date: date) -> date:
    """Return a random date within an inclusive date range."""

    return start_date + timedelta(days=rng.randint(0, (end_date - start_date).days))


def _parse_date(value: str | date) -> date:
    """Parse a stored ISO date value."""

    if isinstance(value, date):
        return value
    return datetime.strptime(value, DATE_FORMAT).date()


def _calculate_purchase_frequency(transaction_count: int, tenure_days: int) -> float:
    """Calculate annualized purchase frequency with a short-tenure floor."""

    tenure_years = max(tenure_days / 365.25, 0.25)
    return round(transaction_count / tenure_years, 3)


def _seasonality_multiplier(transaction_date: date) -> float:
    """Add mild seasonal variation to purchase likelihood."""

    if transaction_date.month in {11, 12}:
        return 1.22
    if transaction_date.month in {1, 2}:
        return 0.92
    if transaction_date.month in {6, 7}:
        return 1.08
    return 1.0


def _round_price(price: float) -> float:
    """Round a product price to a common retail-style amount."""

    return round(max(round(price) - 0.01, 0.01), 2)


def _round_money(value: float) -> float:
    """Round a monetary amount to standard cents."""

    return round(max(value, 0.0), 2)


def _as_bool(value: bool | str) -> bool:
    """Interpret in-memory and CSV-style boolean values."""

    if isinstance(value, bool):
        return value
    return value.lower() == "true"


def estimate_purchase_probability(
    inputs: CustomerBehaviorInputs,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> float:
    """Estimate the probability of purchase in a future observation window.

    Price increases modestly reduce purchase likelihood, discounts partially
    offset price friction, and stronger customer history increases purchase
    likelihood. Region and acquisition channel add realistic baseline variation
    without creating deterministic segments.
    """

    discount_rate = _normalize_discount(inputs.discount_percent)
    purchase_frequency = _normalize_purchase_frequency(inputs.purchase_frequency)
    tenure = _normalize_tenure(inputs.customer_tenure_days)
    prior_spending = _normalize_prior_spending(inputs.prior_spending)

    score = config.purchase_intercept
    if inputs.price_increase_occurred:
        score += config.price_increase_purchase_effect
    score += config.discount_purchase_effect * discount_rate
    score += config.frequency_purchase_effect * purchase_frequency
    score += config.tenure_purchase_effect * tenure
    score += config.spending_purchase_effect * prior_spending
    score += REGION_PURCHASE_EFFECTS.get(inputs.customer_region, 0.0)
    score += ACQUISITION_PURCHASE_EFFECTS.get(inputs.acquisition_channel, 0.0)

    return _clip_probability(_sigmoid(score), config)


def estimate_churn_probability(
    inputs: CustomerBehaviorInputs,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> float:
    """Estimate the probability of customer churn in a future observation window.

    Price increases raise churn risk probabilistically, while discounts,
    frequent purchases, longer tenure, and higher prior spending generally lower
    churn risk. Region and acquisition channel provide non-deterministic
    heterogeneity for later retention and causal analyses.
    """

    discount_rate = _normalize_discount(inputs.discount_percent)
    purchase_frequency = _normalize_purchase_frequency(inputs.purchase_frequency)
    tenure = _normalize_tenure(inputs.customer_tenure_days)
    prior_spending = _normalize_prior_spending(inputs.prior_spending)

    score = config.churn_intercept
    if inputs.price_increase_occurred:
        score += config.price_increase_churn_effect
    score += config.discount_churn_effect * discount_rate
    score += config.frequency_churn_effect * purchase_frequency
    score += config.tenure_churn_effect * tenure
    score += config.spending_churn_effect * prior_spending
    score += REGION_CHURN_EFFECTS.get(inputs.customer_region, 0.0)
    score += ACQUISITION_CHURN_EFFECTS.get(inputs.acquisition_channel, 0.0)

    return _clip_probability(_sigmoid(score), config)


def get_dataset_schema() -> tuple[SchemaField, ...]:
    """Return the schema for the synthetic ecommerce dataset."""

    return SYNTHETIC_DATASET_SCHEMA


def generate_customer_profiles(
    n_customers: int = DEFAULT_CUSTOMER_COUNT,
    rng: Random | None = None,
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> list[dict[str, object]]:
    """Generate customer identifiers, regions, channels, and first purchase dates."""

    rng = rng or Random(RANDOM_SEED)
    latest_first_purchase_date = max(start_date, end_date - timedelta(days=30))
    customers: list[dict[str, object]] = []

    for customer_number in range(1, n_customers + 1):
        customers.append(
            {
                "customer_id": f"C{customer_number:06d}",
                "first_purchase_date": _random_date(
                    rng, start_date, latest_first_purchase_date
                ),
                "customer_region": _weighted_choice(REGION_WEIGHTS, rng),
                "acquisition_channel": _weighted_choice(
                    ACQUISITION_CHANNEL_WEIGHTS, rng
                ),
                "latent_engagement": min(
                    max(rng.lognormvariate(-0.05, 0.45), 0.45), 2.25
                ),
                "price_sensitivity": min(max(rng.gauss(1.0, 0.18), 0.65), 1.45),
                "discount_affinity": min(max(rng.gauss(1.0, 0.22), 0.55), 1.65),
            }
        )

    return customers


def generate_product_catalog(rng: Random | None = None) -> tuple[Product, ...]:
    """Generate a deterministic synthetic product/SKU catalog."""

    rng = rng or Random(RANDOM_SEED + 101)
    products: list[Product] = []
    next_product_number = 1

    for category in PRODUCT_CATEGORY_TEMPLATES:
        product_names = SYNTHETIC_PRODUCT_NAMES[category.name]
        for product_name in product_names:
            list_price = _round_price(
                max(
                    rng.gauss(category.base_price, category.price_stddev * 0.45),
                    category.floor_price,
                )
            )
            cost_rate = rng.uniform(0.34, 0.52)
            unit_cost = _round_money(list_price * cost_rate)
            product_weight = category.weight * rng.uniform(0.78, 1.22) / len(
                product_names
            )
            starting_inventory = int(
                max(180, rng.gauss(560, 110) * (0.72 + product_weight * 24))
            )

            products.append(
                Product(
                    product_id=f"P{next_product_number:04d}",
                    product_name=product_name,
                    product_category=category.name,
                    unit_cost=unit_cost,
                    list_price=list_price,
                    price_stddev=category.price_stddev * 0.16,
                    floor_price=category.floor_price,
                    weight=product_weight,
                    price_increase_tendency=(
                        category.price_increase_tendency + rng.gauss(0.0, 0.025)
                    ),
                    price_elasticity=min(
                        max(category.price_elasticity + rng.gauss(0.0, 0.16), -1.85),
                        -0.55,
                    ),
                    starting_inventory=starting_inventory,
                )
            )
            next_product_number += 1

    return tuple(products)


def assign_pricing_events(
    transaction_date: date,
    product: Product,
    customer_region: str,
    acquisition_channel: str,
    purchase_frequency: float,
    prior_spending: float,
    rng: Random,
) -> bool:
    """Probabilistically mark whether a transaction sees a price increase."""

    if transaction_date < PRICE_EVENT_START_DATE:
        score = -4.25
    else:
        score = -1.45 + product.price_increase_tendency

    score += REGION_PRICE_EXPOSURE_EFFECTS.get(customer_region, 0.0)
    score += ACQUISITION_PRICE_EXPOSURE_EFFECTS.get(acquisition_channel, 0.0)
    score += 0.20 * _normalize_prior_spending(prior_spending)
    score += 0.15 * _normalize_purchase_frequency(purchase_frequency)

    probability = min(max(_sigmoid(score), 0.01), 0.65)
    return rng.random() < probability


def assign_discount_percent(
    price_increase_occurred: bool,
    prior_spending: float,
    acquisition_channel: str,
    discount_affinity: float,
    rng: Random,
) -> float:
    """Probabilistically assign a transaction-level discount percentage."""

    discount_chance = 0.32
    if price_increase_occurred:
        discount_chance += 0.07
    discount_chance += ACQUISITION_DISCOUNT_EFFECTS.get(acquisition_channel, 0.0)
    discount_chance += 0.05 * _normalize_prior_spending(prior_spending)
    discount_chance += 0.08 * (discount_affinity - 1.0)
    discount_chance = min(max(discount_chance, 0.05), 0.72)

    if rng.random() >= discount_chance:
        return 0.0

    return _weighted_float_choice(DISCOUNT_WEIGHTS, rng)


def _product_price_demand_multiplier(
    product: Product,
    list_price: float,
    customer_price_sensitivity: float = 1.0,
) -> float:
    """Estimate product-level demand response to a displayed list price.

    Product/category elasticity introduces a downward-sloping price response:
    higher list prices reduce expected purchase probability on average, while
    seasonality, popularity, discounts, and random variation can still produce
    strong sales on some price-increase days.
    """

    price_index = max(list_price, 0.01) / max(product.list_price, 0.01)
    sensitivity = min(max(customer_price_sensitivity, 0.65), 1.45)
    multiplier = price_index ** (product.price_elasticity * sensitivity)
    return min(max(multiplier, 0.58), 1.32)


def _generate_product_price(
    product: Product,
    price_increase_occurred: bool,
    rng: Random,
) -> float:
    """Generate a positive list price for one synthetic product."""

    price = rng.gauss(product.list_price, product.price_stddev)
    if price_increase_occurred:
        price *= rng.uniform(1.04, 1.14)

    return _round_price(max(price, product.floor_price))


def _product_price_context_for_day(
    product: Product,
    event_date: date,
    price_increase_occurred: bool,
    rng: Random,
) -> ProductPriceContext:
    """Generate product-day price context before demand is realized."""

    list_price = product.list_price
    if price_increase_occurred:
        list_price *= rng.uniform(1.04, 1.12)
    list_price = _round_price(max(list_price, product.floor_price))

    discount_probability = 0.16
    if event_date.month in {11, 12}:
        discount_probability += 0.12
    if price_increase_occurred:
        discount_probability += 0.03
    discount_percent = (
        _weighted_float_choice(DISCOUNT_WEIGHTS, rng)
        if rng.random() < discount_probability
        else 0.0
    )
    selling_price = round(list_price * (1 - _normalize_discount(discount_percent)), 2)

    return ProductPriceContext(
        list_price=list_price,
        selling_price=selling_price,
        discount_percent=discount_percent,
        price_increase_occurred=price_increase_occurred,
    )


def _select_price_increase_dates(
    product: Product,
    rng: Random,
    start_date: date,
    end_date: date,
) -> set[date]:
    """Choose product-day price test windows independently of realized demand."""

    all_dates = _date_range(start_date, end_date)
    price_increase_dates: set[date] = set()
    second_window_probability = min(
        max(_sigmoid(-0.25 + product.price_increase_tendency), 0.28),
        0.68,
    )
    window_count = 1 + int(rng.random() < second_window_probability)
    if rng.random() < max(0.08, second_window_probability - 0.45):
        window_count += 1

    for _ in range(window_count):
        duration = rng.randint(28, 64)
        start_offset = rng.randint(0, max(len(all_dates) - duration, 0))
        for offset in range(start_offset, min(start_offset + duration, len(all_dates))):
            price_increase_dates.add(all_dates[offset])

    # Small holdout-style daily tests add variation without tying price changes
    # to peak sales days or transaction counts.
    for event_date in all_dates:
        if rng.random() < 0.006:
            price_increase_dates.add(event_date)

    return price_increase_dates


def generate_product_price_calendar(
    products: tuple[Product, ...],
    rng: Random | None = None,
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> dict[tuple[str, date], ProductPriceContext]:
    """Generate a product-day price calendar independent of realized demand."""

    rng = rng or Random(RANDOM_SEED + 151)
    calendar: dict[tuple[str, date], ProductPriceContext] = {}
    all_dates = _date_range(start_date, end_date)

    for product in products:
        price_increase_dates = _select_price_increase_dates(
            product,
            rng,
            start_date,
            end_date,
        )
        for event_date in all_dates:
            calendar[(product.product_id, event_date)] = _product_price_context_for_day(
                product,
                event_date,
                event_date in price_increase_dates,
                rng,
            )

    return calendar


def _generate_quantity(discount_percent: float, rng: Random) -> int:
    """Generate a small ecommerce order quantity."""

    discount_rate = _normalize_discount(discount_percent)
    two_item_probability = 0.16 + (0.35 * discount_rate)
    three_item_probability = 0.04 + (0.12 * discount_rate)
    four_item_probability = 0.01 + (0.04 * discount_rate)
    draw = rng.random()

    if draw < four_item_probability:
        return 4
    if draw < four_item_probability + three_item_probability:
        return 3
    if draw < four_item_probability + three_item_probability + two_item_probability:
        return 2
    return 1


def _create_transaction_record(
    customer: dict[str, object],
    transaction_id: str,
    transaction_date: date,
    product: Product,
    price_increase_occurred: bool,
    discount_percent: float,
    transaction_count: int,
    prior_spending: float,
    rng: Random,
    list_price_override: float | None = None,
) -> dict[str, object]:
    """Create one transaction row using current customer state."""

    first_purchase_date = customer["first_purchase_date"]
    if not isinstance(first_purchase_date, date):
        raise TypeError("first_purchase_date must be a date before CSV output.")

    customer_tenure_days = max((transaction_date - first_purchase_date).days, 0)
    purchase_frequency = _calculate_purchase_frequency(
        transaction_count, customer_tenure_days
    )
    list_price = (
        _round_money(list_price_override)
        if list_price_override is not None
        else _generate_product_price(product, price_increase_occurred, rng)
    )
    quantity = _generate_quantity(discount_percent, rng)
    gross_order_value = list_price * quantity
    discount_amount = round(gross_order_value * _normalize_discount(discount_percent), 2)
    order_value = round(gross_order_value - discount_amount, 2)
    selling_price = round(order_value / quantity, 2)
    gross_margin = round(order_value - (product.unit_cost * quantity), 2)

    return {
        "transaction_id": transaction_id,
        "customer_id": customer["customer_id"],
        "transaction_date": transaction_date.isoformat(),
        "product_id": product.product_id,
        "product_name": product.product_name,
        "product_category": product.product_category,
        "product_price": list_price,
        "unit_cost": product.unit_cost,
        "list_price": list_price,
        "selling_price": selling_price,
        "discount_amount": discount_amount,
        "discount_percent": discount_percent,
        "quantity": quantity,
        "order_value": order_value,
        "gross_margin": gross_margin,
        "customer_region": customer["customer_region"],
        "acquisition_channel": customer["acquisition_channel"],
        "customer_tenure_days": customer_tenure_days,
        "purchase_frequency": purchase_frequency,
        "prior_spending": round(prior_spending, 2),
        "price_increase_occurred": price_increase_occurred,
        "churned": False,
    }


def _purchase_probability_for_candidate(
    customer: dict[str, object],
    transaction_date: date,
    product: Product,
    list_price: float,
    price_increase_occurred: bool,
    discount_percent: float,
    transaction_count: int,
    prior_spending: float,
) -> float:
    """Estimate repeat purchase probability for one candidate transaction date."""

    first_purchase_date = customer["first_purchase_date"]
    if not isinstance(first_purchase_date, date):
        raise TypeError("first_purchase_date must be a date before CSV output.")

    customer_tenure_days = max((transaction_date - first_purchase_date).days, 0)
    purchase_frequency = _calculate_purchase_frequency(
        transaction_count, customer_tenure_days
    )
    inputs = CustomerBehaviorInputs(
        price_increase_occurred=price_increase_occurred,
        discount_percent=discount_percent,
        purchase_frequency=purchase_frequency,
        customer_tenure_days=customer_tenure_days,
        prior_spending=prior_spending,
        customer_region=str(customer["customer_region"]),
        acquisition_channel=str(customer["acquisition_channel"]),
    )
    probability = estimate_purchase_probability(inputs)
    probability *= float(customer["latent_engagement"])
    probability *= _seasonality_multiplier(transaction_date)
    probability *= _product_price_demand_multiplier(
        product,
        list_price,
        float(customer["price_sensitivity"]),
    )
    if price_increase_occurred:
        probability *= max(0.70, 1.05 - 0.12 * float(customer["price_sensitivity"]))

    return min(max(probability * 0.42, 0.01), 0.88)


def generate_transaction_history(
    customers: list[dict[str, object]],
    catalog: tuple[Product, ...] | None = None,
    price_calendar: dict[tuple[str, date], ProductPriceContext] | None = None,
    rng: Random | None = None,
    end_date: date = END_DATE,
) -> list[dict[str, object]]:
    """Generate transaction dates, quantities, discounts, and order values."""

    rng = rng or Random(RANDOM_SEED)
    catalog = catalog or generate_product_catalog()
    price_calendar = price_calendar or generate_product_price_calendar(
        catalog,
        rng,
        START_DATE,
        end_date,
    )
    transactions: list[dict[str, object]] = []
    next_transaction_number = 1

    for customer in customers:
        first_purchase_date = customer["first_purchase_date"]
        if not isinstance(first_purchase_date, date):
            raise TypeError("first_purchase_date must be a date before CSV output.")

        transaction_count = 0
        prior_spending = 0.0

        initial_product = _choose_product(catalog, rng)
        initial_price_context = price_calendar[
            (initial_product.product_id, first_purchase_date)
        ]
        initial_price_change = initial_price_context.price_increase_occurred
        initial_discount = max(
            initial_price_context.discount_percent,
            assign_discount_percent(
                initial_price_change,
                prior_spending=0.0,
                acquisition_channel=str(customer["acquisition_channel"]),
                discount_affinity=float(customer["discount_affinity"]),
                rng=rng,
            ),
        )
        initial_transaction = _create_transaction_record(
            customer,
            f"T{next_transaction_number:08d}",
            first_purchase_date,
            initial_product,
            initial_price_change,
            initial_discount,
            transaction_count,
            prior_spending,
            rng,
            list_price_override=initial_price_context.list_price,
        )
        transactions.append(initial_transaction)
        next_transaction_number += 1
        transaction_count += 1
        prior_spending += float(initial_transaction["order_value"])

        candidate_date = first_purchase_date + timedelta(days=rng.randint(21, 45))
        while candidate_date <= end_date:
            product = _choose_product(catalog, rng)
            price_context = price_calendar[(product.product_id, candidate_date)]
            customer_tenure_days = max((candidate_date - first_purchase_date).days, 0)
            purchase_frequency = _calculate_purchase_frequency(
                transaction_count, customer_tenure_days
            )
            price_change = price_context.price_increase_occurred
            discount_percent = max(
                price_context.discount_percent,
                assign_discount_percent(
                    price_change,
                    prior_spending,
                    str(customer["acquisition_channel"]),
                    float(customer["discount_affinity"]),
                    rng,
                ),
            )
            purchase_probability = _purchase_probability_for_candidate(
                customer,
                candidate_date,
                product,
                price_context.list_price,
                price_change,
                discount_percent,
                transaction_count,
                prior_spending,
            )

            if rng.random() < purchase_probability:
                transaction = _create_transaction_record(
                    customer,
                    f"T{next_transaction_number:08d}",
                    candidate_date,
                    product,
                    price_change,
                    discount_percent,
                    transaction_count,
                    prior_spending,
                    rng,
                    list_price_override=price_context.list_price,
                )
                transactions.append(transaction)
                next_transaction_number += 1
                transaction_count += 1
                prior_spending += float(transaction["order_value"])

            candidate_date += timedelta(days=rng.randint(20, 55))

    return transactions


def derive_customer_metrics(
    customers: list[dict[str, object]],
    transactions: list[dict[str, object]],
    end_date: date = END_DATE,
) -> dict[str, dict[str, object]]:
    """Derive purchase frequency, prior spending, and exposure features."""

    metrics: dict[str, dict[str, object]] = {}
    for customer in customers:
        customer_id = str(customer["customer_id"])
        first_purchase_date = customer["first_purchase_date"]
        if not isinstance(first_purchase_date, date):
            raise TypeError("first_purchase_date must be a date before CSV output.")

        metrics[customer_id] = {
            "first_purchase_date": first_purchase_date,
            "customer_tenure_days": max((end_date - first_purchase_date).days, 0),
            "transaction_count": 0,
            "total_spending": 0.0,
            "discount_percent_total": 0.0,
            "price_increase_count": 0,
            "last_transaction_date": first_purchase_date,
        }

    for transaction in transactions:
        customer_id = str(transaction["customer_id"])
        transaction_date = _parse_date(transaction["transaction_date"])
        customer_metrics = metrics[customer_id]
        customer_metrics["transaction_count"] = (
            int(customer_metrics["transaction_count"]) + 1
        )
        customer_metrics["total_spending"] = round(
            float(customer_metrics["total_spending"])
            + float(transaction["order_value"]),
            2,
        )
        customer_metrics["discount_percent_total"] = (
            float(customer_metrics["discount_percent_total"])
            + float(transaction["discount_percent"])
        )
        customer_metrics["price_increase_count"] = (
            int(customer_metrics["price_increase_count"])
            + int(_as_bool(transaction["price_increase_occurred"]))
        )
        if transaction_date > customer_metrics["last_transaction_date"]:
            customer_metrics["last_transaction_date"] = transaction_date

    for customer_metrics in metrics.values():
        transaction_count = int(customer_metrics["transaction_count"])
        tenure_days = int(customer_metrics["customer_tenure_days"])
        customer_metrics["purchase_frequency"] = _calculate_purchase_frequency(
            transaction_count, tenure_days
        )
        customer_metrics["average_discount_percent"] = round(
            float(customer_metrics["discount_percent_total"]) / transaction_count,
            2,
        )
        customer_metrics["price_increase_occurred"] = (
            int(customer_metrics["price_increase_count"]) > 0
        )
        customer_metrics["prior_spending"] = round(
            float(customer_metrics["total_spending"]), 2
        )

    return metrics


def assign_churn_outcomes(
    customers: list[dict[str, object]],
    customer_metrics: dict[str, dict[str, object]],
    rng: Random | None = None,
) -> list[dict[str, object]]:
    """Assign churn labels using probabilistic behavioral logic."""

    rng = rng or Random(RANDOM_SEED)
    customer_rows: list[dict[str, object]] = []

    for customer in customers:
        customer_id = str(customer["customer_id"])
        metrics = customer_metrics[customer_id]
        inputs = CustomerBehaviorInputs(
            price_increase_occurred=bool(metrics["price_increase_occurred"]),
            discount_percent=float(metrics["average_discount_percent"]),
            purchase_frequency=float(metrics["purchase_frequency"]),
            customer_tenure_days=int(metrics["customer_tenure_days"]),
            prior_spending=float(metrics["prior_spending"]),
            customer_region=str(customer["customer_region"]),
            acquisition_channel=str(customer["acquisition_channel"]),
        )
        churn_probability = estimate_churn_probability(inputs)
        churned = rng.random() < churn_probability

        customer_rows.append(
            {
                "customer_id": customer_id,
                "first_purchase_date": metrics["first_purchase_date"].isoformat(),
                "customer_region": customer["customer_region"],
                "acquisition_channel": customer["acquisition_channel"],
                "customer_tenure_days": metrics["customer_tenure_days"],
                "purchase_frequency": metrics["purchase_frequency"],
                "prior_spending": metrics["prior_spending"],
                "transaction_count": metrics["transaction_count"],
                "total_spending": metrics["total_spending"],
                "average_discount_percent": metrics["average_discount_percent"],
                "price_increase_occurred": metrics["price_increase_occurred"],
                "churned": churned,
                "last_transaction_date": metrics["last_transaction_date"].isoformat(),
            }
        )

    return customer_rows


def _apply_churn_to_transactions(
    transactions: list[dict[str, object]],
    customers: list[dict[str, object]],
) -> None:
    """Attach final customer churn labels to each transaction row."""

    churn_lookup = {
        str(customer["customer_id"]): bool(customer["churned"]) for customer in customers
    }
    for transaction in transactions:
        transaction["churned"] = churn_lookup[str(transaction["customer_id"])]


def _date_range(start_date: date, end_date: date) -> list[date]:
    """Return all dates in an inclusive date range."""

    return [
        start_date + timedelta(days=day_offset)
        for day_offset in range((end_date - start_date).days + 1)
    ]


def _aggregate_transactions_by_product_date(
    transactions: list[dict[str, object]],
) -> dict[tuple[str, date], dict[str, float]]:
    """Aggregate transaction economics by product and transaction date."""

    aggregates: dict[tuple[str, date], dict[str, float]] = {}
    for transaction in transactions:
        product_id = str(transaction["product_id"])
        transaction_date = _parse_date(transaction["transaction_date"])
        key = (product_id, transaction_date)
        aggregate = aggregates.setdefault(
            key,
            {
                "purchases": 0.0,
                "units_sold": 0.0,
                "revenue": 0.0,
                "gross_margin": 0.0,
                "list_price_total": 0.0,
                "selling_price_total": 0.0,
                "discount_percent_total": 0.0,
                "price_increase_count": 0.0,
            },
        )

        quantity = int(transaction["quantity"])
        aggregate["purchases"] += 1
        aggregate["units_sold"] += quantity
        aggregate["revenue"] += float(transaction["order_value"])
        aggregate["gross_margin"] += float(transaction["gross_margin"])
        aggregate["list_price_total"] += float(transaction["list_price"]) * quantity
        aggregate["selling_price_total"] += float(transaction["selling_price"]) * quantity
        aggregate["discount_percent_total"] += float(transaction["discount_percent"])
        aggregate["price_increase_count"] += int(
            _as_bool(transaction["price_increase_occurred"])
        )

    return aggregates


def _daily_product_price_context(
    product: Product,
    event_date: date,
    rng: Random,
) -> tuple[float, float, float, bool]:
    """Generate price context for product-days without observed purchases."""

    if event_date < PRICE_EVENT_START_DATE:
        price_increase_probability = 0.02
    else:
        price_increase_probability = min(
            max(_sigmoid(-1.65 + product.price_increase_tendency), 0.03),
            0.48,
        )
    price_increase_occurred = rng.random() < price_increase_probability
    list_price = product.list_price
    if price_increase_occurred:
        list_price *= rng.uniform(1.04, 1.12)
    list_price = _round_price(max(list_price, product.floor_price))

    discount_probability = 0.16
    if event_date.month in {11, 12}:
        discount_probability += 0.12
    if price_increase_occurred:
        discount_probability += 0.04
    discount_percent = (
        _weighted_float_choice(DISCOUNT_WEIGHTS, rng)
        if rng.random() < discount_probability
        else 0.0
    )
    selling_price = round(list_price * (1 - _normalize_discount(discount_percent)), 2)
    return list_price, selling_price, discount_percent, price_increase_occurred


def _estimate_product_funnel_counts(
    product: Product,
    event_date: date,
    selling_price: float,
    discount_percent: float,
    purchases: int,
    stockout_flag: bool,
    rng: Random,
) -> tuple[int, int, int]:
    """Generate product views, add-to-cart events, and checkout-start events."""

    discount_rate = _normalize_discount(discount_percent)
    price_index = selling_price / max(product.list_price, 0.01)
    seasonality = _seasonality_multiplier(event_date)
    views = int(
        max(
            24,
            rng.gauss(1.0, 0.16)
            * seasonality
            * (42 + product.weight * 5_800)
            * (1 + 0.55 * discount_rate)
            * max(0.62, 1.08 - 0.20 * (price_index - 1)),
        )
    )

    if stockout_flag:
        views = int(views * rng.uniform(0.45, 0.72))

    purchase_conversion_rate = min(max(0.012 + (0.034 * discount_rate), 0.008), 0.07)
    if purchases > 0:
        views = max(views, int(purchases / purchase_conversion_rate) + rng.randint(8, 34))

    add_to_cart_rate = min(
        max(0.08 + (0.24 * discount_rate) - (0.025 * (price_index - 1)), 0.04),
        0.34,
    )
    add_to_cart_events = int(round(views * add_to_cart_rate * rng.uniform(0.88, 1.14)))
    add_to_cart_events = min(max(add_to_cart_events, purchases), views)

    checkout_rate = min(max(0.38 + (0.18 * discount_rate), 0.24), 0.76)
    checkout_started_events = int(
        round(add_to_cart_events * checkout_rate * rng.uniform(0.90, 1.12))
    )
    checkout_started_events = min(
        max(checkout_started_events, purchases),
        add_to_cart_events,
    )

    return views, add_to_cart_events, checkout_started_events


def generate_product_event_history(
    products: tuple[Product, ...],
    transactions: list[dict[str, object]],
    price_calendar: dict[tuple[str, date], ProductPriceContext] | None = None,
    rng: Random | None = None,
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> list[dict[str, object]]:
    """Generate product-day funnel, price, margin, and inventory event rows."""

    rng = rng or Random(RANDOM_SEED + 202)
    transaction_aggregates = _aggregate_transactions_by_product_date(transactions)
    product_events: list[dict[str, object]] = []

    for product in products:
        inventory_level = product.starting_inventory
        reorder_point = max(35, int(product.starting_inventory * 0.10))
        # The stockout flag marks materially constrained availability, such as
        # limited sizes or colors, rather than only literal zero inventory.
        stockout_threshold = max(70, int(product.starting_inventory * 0.22))
        for event_date in _date_range(start_date, end_date):
            if event_date.day == 1:
                inventory_level += int(
                    max(20, product.starting_inventory * rng.uniform(0.035, 0.08))
                )
            if inventory_level < reorder_point and rng.random() < 0.30:
                inventory_level += int(
                    max(45, product.starting_inventory * rng.uniform(0.12, 0.26))
                )

            price_context = (
                price_calendar.get((product.product_id, event_date))
                if price_calendar is not None
                else None
            )
            aggregate = transaction_aggregates.get((product.product_id, event_date))
            if aggregate:
                purchases = int(aggregate["purchases"])
                units_sold = int(aggregate["units_sold"])
                revenue = round(aggregate["revenue"], 2)
                gross_margin = round(aggregate["gross_margin"], 2)
                list_price = round(
                    aggregate["list_price_total"] / max(units_sold, 1),
                    2,
                )
                selling_price = round(
                    aggregate["selling_price_total"] / max(units_sold, 1),
                    2,
                )
                discount_percent = round(
                    aggregate["discount_percent_total"] / max(purchases, 1),
                    2,
                )
                price_increase_occurred = (
                    price_context.price_increase_occurred
                    if price_context is not None
                    else aggregate["price_increase_count"] > 0
                )
            else:
                purchases = 0
                units_sold = 0
                revenue = 0.0
                gross_margin = 0.0
                if price_context is None:
                    (
                        list_price,
                        selling_price,
                        discount_percent,
                        price_increase_occurred,
                    ) = _daily_product_price_context(product, event_date, rng)
                else:
                    list_price = price_context.list_price
                    selling_price = price_context.selling_price
                    discount_percent = price_context.discount_percent
                    price_increase_occurred = price_context.price_increase_occurred

            inventory_level = max(inventory_level - units_sold, 0)
            stockout_flag = inventory_level <= stockout_threshold
            (
                product_views,
                add_to_cart_events,
                checkout_started_events,
            ) = _estimate_product_funnel_counts(
                product,
                event_date,
                selling_price,
                discount_percent,
                purchases,
                stockout_flag,
                rng,
            )

            product_events.append(
                {
                    "event_date": event_date.isoformat(),
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "product_category": product.product_category,
                    "unit_cost": product.unit_cost,
                    "list_price": list_price,
                    "selling_price": selling_price,
                    "discount_percent": discount_percent,
                    "product_views": product_views,
                    "add_to_cart_events": add_to_cart_events,
                    "checkout_started_events": checkout_started_events,
                    "purchases": purchases,
                    "units_sold": units_sold,
                    "revenue": revenue,
                    "gross_margin": gross_margin,
                    "inventory_level": inventory_level,
                    "stockout_flag": stockout_flag,
                    "price_increase_occurred": price_increase_occurred,
                }
            )

    return product_events


def derive_product_records(
    products: tuple[Product, ...],
    product_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate product-day event data into product-level raw records."""

    product_lookup = {product.product_id: product for product in products}
    metrics: dict[str, dict[str, object]] = {
        product.product_id: {
            "total_revenue": 0.0,
            "total_units_sold": 0,
            "total_purchases": 0,
            "product_views": 0,
            "add_to_cart_events": 0,
            "checkout_started_events": 0,
            "gross_margin": 0.0,
            "inventory_level": product.starting_inventory,
            "stockout_flag": False,
        }
        for product in products
    }

    for event in product_events:
        product_id = str(event["product_id"])
        product_metrics = metrics[product_id]
        product_metrics["total_revenue"] = round(
            float(product_metrics["total_revenue"]) + float(event["revenue"]),
            2,
        )
        product_metrics["total_units_sold"] = (
            int(product_metrics["total_units_sold"]) + int(event["units_sold"])
        )
        product_metrics["total_purchases"] = (
            int(product_metrics["total_purchases"]) + int(event["purchases"])
        )
        product_metrics["product_views"] = (
            int(product_metrics["product_views"]) + int(event["product_views"])
        )
        product_metrics["add_to_cart_events"] = (
            int(product_metrics["add_to_cart_events"])
            + int(event["add_to_cart_events"])
        )
        product_metrics["checkout_started_events"] = (
            int(product_metrics["checkout_started_events"])
            + int(event["checkout_started_events"])
        )
        product_metrics["gross_margin"] = round(
            float(product_metrics["gross_margin"]) + float(event["gross_margin"]),
            2,
        )
        product_metrics["inventory_level"] = int(event["inventory_level"])
        product_metrics["stockout_flag"] = bool(
            product_metrics["stockout_flag"]
        ) or _as_bool(event["stockout_flag"])

    product_rows: list[dict[str, object]] = []
    for product_id, product_metrics in metrics.items():
        product = product_lookup[product_id]
        total_revenue = float(product_metrics["total_revenue"])
        gross_margin = float(product_metrics["gross_margin"])
        gross_margin_rate = round(gross_margin / total_revenue, 4) if total_revenue else 0.0
        product_rows.append(
            {
                "product_id": product.product_id,
                "product_name": product.product_name,
                "product_category": product.product_category,
                "unit_cost": product.unit_cost,
                "list_price": product.list_price,
                "gross_margin": gross_margin,
                "gross_margin_rate": gross_margin_rate,
                "total_revenue": total_revenue,
                "total_units_sold": product_metrics["total_units_sold"],
                "total_purchases": product_metrics["total_purchases"],
                "product_views": product_metrics["product_views"],
                "add_to_cart_events": product_metrics["add_to_cart_events"],
                "checkout_started_events": product_metrics["checkout_started_events"],
                "inventory_level": product_metrics["inventory_level"],
                "stockout_flag": product_metrics["stockout_flag"],
            }
        )

    return product_rows


def validate_customer_records(
    customers: list[dict[str, object]],
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> None:
    """Validate customer-level raw records for impossible or missing values."""

    seen_customer_ids: set[str] = set()
    for row in customers:
        customer_id = str(row.get("customer_id", ""))
        if not customer_id:
            raise ValueError("Customer record has a missing customer_id.")
        if customer_id in seen_customer_ids:
            raise ValueError(f"Duplicate customer_id found: {customer_id}")
        seen_customer_ids.add(customer_id)

        first_purchase_date = _parse_date(row["first_purchase_date"])
        last_transaction_date = _parse_date(row["last_transaction_date"])
        if first_purchase_date < start_date or first_purchase_date > end_date:
            raise ValueError(f"Invalid first purchase date for {customer_id}.")
        if last_transaction_date < start_date or last_transaction_date > end_date:
            raise ValueError(f"Invalid last transaction date for {customer_id}.")
        if last_transaction_date < first_purchase_date:
            raise ValueError(f"Last transaction predates first purchase for {customer_id}.")
        if int(row["customer_tenure_days"]) < 0:
            raise ValueError(f"Negative tenure found for {customer_id}.")
        if int(row["transaction_count"]) <= 0:
            raise ValueError(f"Impossible transaction count found for {customer_id}.")

        for field in (
            "purchase_frequency",
            "prior_spending",
            "total_spending",
            "average_discount_percent",
        ):
            if float(row[field]) < 0:
                raise ValueError(f"Negative {field} found for {customer_id}.")


def validate_transaction_records(
    transactions: list[dict[str, object]],
    customer_ids: set[str],
    product_ids: set[str],
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> None:
    """Validate transaction-level raw records for impossible or missing values."""

    seen_transaction_ids: set[str] = set()
    for row in transactions:
        transaction_id = str(row.get("transaction_id", ""))
        customer_id = str(row.get("customer_id", ""))
        product_id = str(row.get("product_id", ""))
        if not transaction_id:
            raise ValueError("Transaction record has a missing transaction_id.")
        if not customer_id:
            raise ValueError(f"Transaction {transaction_id} has a missing customer_id.")
        if not product_id:
            raise ValueError(f"Transaction {transaction_id} has a missing product_id.")
        if transaction_id in seen_transaction_ids:
            raise ValueError(f"Duplicate transaction_id found: {transaction_id}")
        if customer_id not in customer_ids:
            raise ValueError(f"Unknown customer_id found: {customer_id}")
        if product_id not in product_ids:
            raise ValueError(f"Unknown product_id found: {product_id}")
        seen_transaction_ids.add(transaction_id)

        transaction_date = _parse_date(row["transaction_date"])
        if transaction_date < start_date or transaction_date > end_date:
            raise ValueError(f"Invalid transaction date for {transaction_id}.")
        if not str(row.get("product_name", "")):
            raise ValueError(f"Missing product name found for {transaction_id}.")
        if not str(row.get("product_category", "")):
            raise ValueError(f"Missing product category found for {transaction_id}.")
        if float(row["product_price"]) < 0:
            raise ValueError(f"Negative product price found for {transaction_id}.")
        if float(row["unit_cost"]) <= 0:
            raise ValueError(f"Impossible unit cost found for {transaction_id}.")
        if float(row["list_price"]) <= 0:
            raise ValueError(f"Impossible list price found for {transaction_id}.")
        if float(row["selling_price"]) < 0:
            raise ValueError(f"Negative selling price found for {transaction_id}.")
        if float(row["discount_amount"]) < 0:
            raise ValueError(f"Negative discount amount found for {transaction_id}.")
        if float(row["discount_percent"]) < 0 or float(row["discount_percent"]) > 80:
            raise ValueError(f"Invalid discount percent found for {transaction_id}.")
        if int(row["quantity"]) <= 0:
            raise ValueError(f"Impossible quantity found for {transaction_id}.")
        if float(row["order_value"]) < 0:
            raise ValueError(f"Negative order value found for {transaction_id}.")
        if float(row["gross_margin"]) < 0:
            raise ValueError(f"Negative gross margin found for {transaction_id}.")
        if int(row["customer_tenure_days"]) < 0:
            raise ValueError(f"Negative transaction tenure found for {transaction_id}.")
        if float(row["purchase_frequency"]) < 0:
            raise ValueError(f"Negative purchase frequency found for {transaction_id}.")
        if float(row["prior_spending"]) < 0:
            raise ValueError(f"Negative prior spending found for {transaction_id}.")

        gross_order_value = float(row["product_price"]) * int(row["quantity"])
        if float(row["discount_amount"]) > gross_order_value:
            raise ValueError(f"Discount exceeds gross value for {transaction_id}.")


def validate_product_records(products: list[dict[str, object]]) -> None:
    """Validate product-level raw records for missing IDs and impossible values."""

    seen_product_ids: set[str] = set()
    seen_product_names: set[str] = set()
    for row in products:
        product_id = str(row.get("product_id", ""))
        product_name = str(row.get("product_name", ""))
        if not product_id:
            raise ValueError("Product record has a missing product_id.")
        if product_id in seen_product_ids:
            raise ValueError(f"Duplicate product_id found: {product_id}")
        if not product_name:
            raise ValueError(f"Product {product_id} has a missing product_name.")
        if product_name in seen_product_names:
            raise ValueError(f"Duplicate product_name found: {product_name}")
        if not str(row.get("product_category", "")):
            raise ValueError(f"Product {product_id} has a missing product_category.")
        seen_product_ids.add(product_id)
        seen_product_names.add(product_name)

        for field in (
            "unit_cost",
            "list_price",
            "gross_margin",
            "gross_margin_rate",
            "total_revenue",
            "total_units_sold",
            "total_purchases",
            "product_views",
            "add_to_cart_events",
            "checkout_started_events",
            "inventory_level",
        ):
            if float(row[field]) < 0:
                raise ValueError(f"Negative {field} found for product {product_id}.")
        if float(row["unit_cost"]) > float(row["list_price"]):
            raise ValueError(f"Unit cost exceeds list price for product {product_id}.")
        if int(row["add_to_cart_events"]) > int(row["product_views"]):
            raise ValueError(f"Add-to-cart events exceed views for {product_id}.")
        if int(row["checkout_started_events"]) > int(row["add_to_cart_events"]):
            raise ValueError(f"Checkout starts exceed add-to-cart events for {product_id}.")
        if int(row["total_purchases"]) > int(row["checkout_started_events"]):
            raise ValueError(f"Purchases exceed checkout starts for {product_id}.")


def validate_product_event_records(
    product_events: list[dict[str, object]],
    product_ids: set[str],
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> None:
    """Validate product-day funnel, price, margin, and inventory event rows."""

    seen_event_keys: set[tuple[str, date]] = set()
    for row in product_events:
        product_id = str(row.get("product_id", ""))
        if not product_id:
            raise ValueError("Product event has a missing product_id.")
        if product_id not in product_ids:
            raise ValueError(f"Unknown product_id found in events: {product_id}")
        if not str(row.get("product_name", "")):
            raise ValueError(f"Product event has a missing product_name for {product_id}.")
        if not str(row.get("product_category", "")):
            raise ValueError(f"Product event has a missing category for {product_id}.")

        event_date = _parse_date(row["event_date"])
        event_key = (product_id, event_date)
        if event_key in seen_event_keys:
            raise ValueError(f"Duplicate product event found: {product_id} {event_date}")
        seen_event_keys.add(event_key)
        if event_date < start_date or event_date > end_date:
            raise ValueError(f"Invalid product event date for {product_id}.")

        for field in (
            "unit_cost",
            "list_price",
            "selling_price",
            "discount_percent",
            "product_views",
            "add_to_cart_events",
            "checkout_started_events",
            "purchases",
            "units_sold",
            "revenue",
            "gross_margin",
            "inventory_level",
        ):
            if float(row[field]) < 0:
                raise ValueError(f"Negative {field} found for {product_id}.")
        if float(row["unit_cost"]) > float(row["list_price"]):
            raise ValueError(f"Unit cost exceeds list price for product event {product_id}.")
        if float(row["discount_percent"]) > 80:
            raise ValueError(f"Invalid event discount percent for {product_id}.")
        if int(row["add_to_cart_events"]) > int(row["product_views"]):
            raise ValueError(f"Event add-to-cart exceeds views for {product_id}.")
        if int(row["checkout_started_events"]) > int(row["add_to_cart_events"]):
            raise ValueError(f"Event checkout starts exceed add-to-cart for {product_id}.")
        if int(row["purchases"]) > int(row["checkout_started_events"]):
            raise ValueError(f"Event purchases exceed checkout starts for {product_id}.")
        if int(row["purchases"]) > 0 and int(row["units_sold"]) < int(row["purchases"]):
            raise ValueError(f"Units sold are below purchases for {product_id}.")


def save_raw_outputs(
    customers: list[dict[str, object]],
    transactions: list[dict[str, object]],
    products: list[dict[str, object]],
    product_events: list[dict[str, object]],
    output_dir: Path = RAW_DATA_DIR,
) -> tuple[Path, Path, Path, Path]:
    """Save customer, transaction, product, and event outputs into raw data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    customers_path = output_dir / CUSTOMERS_FILE.name
    transactions_path = output_dir / TRANSACTIONS_FILE.name
    products_path = output_dir / PRODUCTS_FILE.name
    product_events_path = output_dir / PRODUCT_EVENTS_FILE.name

    with customers_path.open("w", newline="", encoding="utf-8") as customer_file:
        writer = csv.DictWriter(customer_file, fieldnames=CUSTOMER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(customers)

    with transactions_path.open("w", newline="", encoding="utf-8") as transaction_file:
        writer = csv.DictWriter(transaction_file, fieldnames=TRANSACTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(transactions)

    with products_path.open("w", newline="", encoding="utf-8") as product_file:
        writer = csv.DictWriter(product_file, fieldnames=PRODUCT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(products)

    with product_events_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as product_event_file:
        writer = csv.DictWriter(
            product_event_file,
            fieldnames=PRODUCT_EVENT_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(product_events)

    return customers_path, transactions_path, products_path, product_events_path


def compare_price_increase_product_days(
    product_events: list[dict[str, object]],
) -> PriceDemandComparison:
    """Compare price and demand on normal versus price-increase product-days."""

    grouped: dict[bool, dict[str, float]] = {
        False: {
            "days": 0.0,
            "selling_price_total": 0.0,
            "units_sold": 0.0,
            "revenue": 0.0,
        },
        True: {
            "days": 0.0,
            "selling_price_total": 0.0,
            "units_sold": 0.0,
            "revenue": 0.0,
        },
    }

    for row in product_events:
        price_increase_occurred = _as_bool(row["price_increase_occurred"])
        metrics = grouped[price_increase_occurred]
        metrics["days"] += 1
        metrics["selling_price_total"] += float(row["selling_price"])
        metrics["units_sold"] += int(row["units_sold"])
        metrics["revenue"] += float(row["revenue"])

    normal = grouped[False]
    increased = grouped[True]
    normal_days = max(normal["days"], 1.0)
    increased_days = max(increased["days"], 1.0)
    normal_average_selling_price = normal["selling_price_total"] / normal_days
    increased_average_selling_price = increased["selling_price_total"] / increased_days
    normal_unit_velocity = normal["units_sold"] / normal_days
    increased_unit_velocity = increased["units_sold"] / increased_days
    normal_revenue_per_day = normal["revenue"] / normal_days
    increased_revenue_per_day = increased["revenue"] / increased_days

    return PriceDemandComparison(
        normal_average_selling_price=normal_average_selling_price,
        increased_average_selling_price=increased_average_selling_price,
        average_price_change=(
            increased_average_selling_price / normal_average_selling_price - 1
        ),
        normal_unit_velocity=normal_unit_velocity,
        increased_unit_velocity=increased_unit_velocity,
        unit_velocity_change=increased_unit_velocity / normal_unit_velocity - 1,
        normal_revenue_per_day=normal_revenue_per_day,
        increased_revenue_per_day=increased_revenue_per_day,
        revenue_per_day_change=increased_revenue_per_day / normal_revenue_per_day - 1,
    )


def summarize_dataset(
    customers: list[dict[str, object]],
    transactions: list[dict[str, object]],
    products: list[dict[str, object]],
    product_events: list[dict[str, object]],
    customers_path: Path,
    transactions_path: Path,
    products_path: Path,
    product_events_path: Path,
) -> GenerationSummary:
    """Calculate a concise summary of the generated raw dataset."""

    transaction_dates = [_parse_date(row["transaction_date"]) for row in transactions]
    average_order_value = mean(float(row["order_value"]) for row in transactions)
    total_revenue = sum(float(row["order_value"]) for row in transactions)
    total_gross_margin = sum(float(row["gross_margin"]) for row in transactions)
    total_product_views = sum(int(row["product_views"]) for row in product_events)
    total_add_to_cart = sum(int(row["add_to_cart_events"]) for row in product_events)
    total_checkout_started = sum(
        int(row["checkout_started_events"]) for row in product_events
    )
    total_product_purchases = sum(int(row["purchases"]) for row in product_events)
    churn_rate = mean(int(_as_bool(row["churned"])) for row in customers)
    price_change_frequency = mean(
        int(_as_bool(row["price_increase_occurred"])) for row in transactions
    )
    stockout_event_rate = mean(
        int(_as_bool(row["stockout_flag"])) for row in product_events
    )
    price_demand_comparison = compare_price_increase_product_days(product_events)

    return GenerationSummary(
        customer_rows=len(customers),
        transaction_rows=len(transactions),
        product_rows=len(products),
        product_event_rows=len(product_events),
        start_date=min(transaction_dates),
        end_date=max(transaction_dates),
        average_order_value=average_order_value,
        average_gross_margin_rate=total_gross_margin / total_revenue,
        churn_rate=churn_rate,
        price_change_frequency=price_change_frequency,
        total_product_views=total_product_views,
        add_to_cart_rate=total_add_to_cart / total_product_views,
        checkout_start_rate=total_checkout_started / total_add_to_cart,
        product_purchase_rate=total_product_purchases / total_checkout_started,
        stockout_event_rate=stockout_event_rate,
        price_demand_comparison=price_demand_comparison,
        customers_path=customers_path,
        transactions_path=transactions_path,
        products_path=products_path,
        product_events_path=product_events_path,
    )


def print_generation_summary(summary: GenerationSummary) -> None:
    """Print a short business-facing summary of the generated dataset."""

    print("Synthetic dataset generated")
    print(f"Customer rows: {summary.customer_rows:,}")
    print(f"Transaction rows: {summary.transaction_rows:,}")
    print(f"Product rows: {summary.product_rows:,}")
    print(f"Product event rows: {summary.product_event_rows:,}")
    print(f"Date range: {summary.start_date} to {summary.end_date}")
    print(f"Average order value: ${summary.average_order_value:,.2f}")
    print(f"Average gross margin rate: {summary.average_gross_margin_rate:.1%}")
    print(f"Churn rate: {summary.churn_rate:.1%}")
    print(f"Price-change frequency: {summary.price_change_frequency:.1%}")
    print("Product-field validation summary")
    print(f"Product views: {summary.total_product_views:,}")
    print(f"Add-to-cart rate: {summary.add_to_cart_rate:.1%}")
    print(f"Checkout-start rate: {summary.checkout_start_rate:.1%}")
    print(f"Purchase-after-checkout rate: {summary.product_purchase_rate:.1%}")
    print(f"Stockout product-day rate: {summary.stockout_event_rate:.1%}")
    comparison = summary.price_demand_comparison
    print("Product price-demand comparison")
    print(
        "Normal product-days: "
        f"avg selling price ${comparison.normal_average_selling_price:,.2f}, "
        f"unit velocity {comparison.normal_unit_velocity:.2f}, "
        f"revenue/day ${comparison.normal_revenue_per_day:,.2f}"
    )
    print(
        "Price-increase product-days: "
        f"avg selling price ${comparison.increased_average_selling_price:,.2f}, "
        f"unit velocity {comparison.increased_unit_velocity:.2f}, "
        f"revenue/day ${comparison.increased_revenue_per_day:,.2f}"
    )
    print(
        "Change on price-increase product-days: "
        f"average price {comparison.average_price_change:+.1%}, "
        f"unit velocity {comparison.unit_velocity_change:+.1%}, "
        f"revenue/day {comparison.revenue_per_day_change:+.1%}"
    )
    print(f"Customer output: {summary.customers_path}")
    print(f"Transaction output: {summary.transactions_path}")
    print(f"Product output: {summary.products_path}")
    print(f"Product event output: {summary.product_events_path}")


def build_synthetic_dataset(
    n_customers: int = DEFAULT_CUSTOMER_COUNT,
    seed: int = RANDOM_SEED,
    output_dir: Path = RAW_DATA_DIR,
) -> GenerationSummary:
    """Orchestrate synthetic data generation, validation, saving, and summary."""

    rng = Random(seed)
    customers = generate_customer_profiles(n_customers=n_customers, rng=rng)
    product_catalog = generate_product_catalog(rng)
    price_calendar = generate_product_price_calendar(product_catalog, rng)
    transactions = generate_transaction_history(
        customers=customers,
        catalog=product_catalog,
        price_calendar=price_calendar,
        rng=rng,
    )
    customer_metrics = derive_customer_metrics(customers, transactions)
    customer_rows = assign_churn_outcomes(customers, customer_metrics, rng)
    _apply_churn_to_transactions(transactions, customer_rows)
    product_events = generate_product_event_history(
        products=product_catalog,
        transactions=transactions,
        price_calendar=price_calendar,
        rng=rng,
    )
    product_rows = derive_product_records(product_catalog, product_events)

    customer_ids = {str(customer["customer_id"]) for customer in customer_rows}
    product_ids = {product.product_id for product in product_catalog}
    validate_customer_records(customer_rows)
    validate_transaction_records(transactions, customer_ids, product_ids)
    validate_product_event_records(product_events, product_ids)
    validate_product_records(product_rows)

    (
        customers_path,
        transactions_path,
        products_path,
        product_events_path,
    ) = save_raw_outputs(
        customers=customer_rows,
        transactions=transactions,
        products=product_rows,
        product_events=product_events,
        output_dir=output_dir,
    )
    summary = summarize_dataset(
        customers=customer_rows,
        transactions=transactions,
        products=product_rows,
        product_events=product_events,
        customers_path=customers_path,
        transactions_path=transactions_path,
        products_path=products_path,
        product_events_path=product_events_path,
    )
    print_generation_summary(summary)

    return summary


def main() -> None:
    """Generate the first complete raw synthetic dataset."""

    build_synthetic_dataset()


if __name__ == "__main__":
    main()
