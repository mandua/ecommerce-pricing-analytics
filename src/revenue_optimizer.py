"""Product-level revenue optimization for the synthetic ecommerce project.

The optimizer estimates how alternative prices may affect demand, revenue,
gross margin, and customer retention for an individual product. Product demand
is driven primarily by price elasticity estimated from the synthetic product
event data. Customer churn is included only as a secondary retention adjustment
using the causal estimate from the causal inference notebook.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PRODUCTS_PATH = RAW_DATA_DIR / "products.csv"
PRODUCT_EVENTS_PATH = RAW_DATA_DIR / "product_events.csv"
TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions.csv"
CUSTOMERS_PATH = RAW_DATA_DIR / "customers.csv"


DEFAULT_CATEGORY_ELASTICITIES = {
    "leggings": -1.20,
    "sports_bras": -1.10,
    "training_tops": -1.25,
    "hoodies": -0.85,
    "joggers": -1.05,
    "shorts": -1.35,
    "outerwear": -0.75,
    "accessories": -1.45,
}


@dataclass(frozen=True)
class OptimizerConfig:
    """Transparent assumptions used by the product-level optimizer."""

    causal_churn_effect: float = 0.0245
    causal_effect_ci_lower: float = 0.0131
    causal_effect_ci_upper: float = 0.0369
    naive_churn_difference: float = 0.0216
    default_reference_price_change: float = 0.07
    min_price_multiplier: float = 0.85
    max_price_multiplier: float = 1.15
    price_grid_step_pct: float = 0.01
    min_elasticity: float = -2.50
    max_elasticity: float = -0.35
    product_elasticity_weight: float = 0.45
    category_elasticity_weight: float = 0.55
    min_elasticity_rows: int = 30
    min_local_response_change: float = 0.05
    max_local_response_change: float = 0.10
    extrapolated_response_weight: float = 0.25
    demand_floor_multiplier: float = 0.25
    demand_ceiling_multiplier: float = 2.50
    price_decrease_churn_relief_factor: float = 0.35
    max_abs_churn_rate_change: float = 0.10
    min_retention_adjustment_multiplier: float = 0.75
    max_retention_adjustment_multiplier: float = 1.08
    min_margin_price_cost_multiple: float = 1.02


@dataclass(frozen=True)
class ProductBaseline:
    """Observed product baseline used as the starting point for simulations."""

    product_id: str
    product_name: str
    product_category: str
    unit_cost: float
    current_average_selling_price: float
    baseline_unit_velocity: float
    baseline_units_sold: float
    baseline_revenue: float
    baseline_gross_margin: float
    baseline_gross_margin_rate: float
    observed_product_days: int
    product_views: int
    purchases: int
    stockout_day_rate: float
    price_increase_day_rate: float
    unique_customers: int
    baseline_product_churn_rate: float
    estimated_price_elasticity: float
    category_price_elasticity: float
    min_tested_price: float
    max_tested_price: float


@dataclass(frozen=True)
class ChurnImpact:
    """Secondary customer-retention impact for a proposed price scenario."""

    baseline_churn_rate: float
    estimated_churn_rate: float
    churn_rate_change: float
    estimated_incremental_churned_customers: float
    retention_demand_multiplier: float
    causal_effect_basis: str


@dataclass(frozen=True)
class ProductPriceScenarioResult:
    """Scenario result for an individual product and proposed selling price."""

    product_id: str
    product_name: str
    product_category: str
    scenario_label: str
    proposed_price: float
    current_average_selling_price: float
    price_change_pct: float
    estimated_price_elasticity: float
    baseline_unit_velocity: float
    expected_unit_velocity: float
    expected_units_sold: float
    expected_revenue: float
    expected_gross_margin: float
    revenue_change_vs_baseline: float
    margin_change_vs_baseline: float
    demand_multiplier: float
    churn_impact: ChurnImpact


@dataclass(frozen=True)
class ProductOptimizationResult:
    """Revenue and margin optima for the tested local price grid."""

    product_id: str
    product_name: str
    product_category: str
    revenue_optimal_scenario: ProductPriceScenarioResult
    margin_optimal_scenario: ProductPriceScenarioResult
    revenue_optimum_boundary_limited: bool
    margin_optimum_boundary_limited: bool
    min_tested_price: float
    max_tested_price: float
    tested_price_count: int


@dataclass(frozen=True)
class PriceOptimumComparison:
    """Compact economics for a secondary tested-price optimum."""

    price: float
    percent_change: float
    expected_units: float
    expected_revenue: float
    expected_gross_margin: float
    revenue_delta: float
    margin_delta: float
    boundary_limited: bool
    boundary_hit: str


@dataclass(frozen=True)
class ProductPriceRecommendation:
    """Dashboard-ready recommendation using gross-margin optimization first."""

    product_id: str
    product_name: str
    product_category: str
    current_price: float
    recommended_price: float
    recommended_percent_change: float
    expected_units: float
    expected_revenue: float
    expected_gross_margin: float
    revenue_delta: float
    margin_delta: float
    estimated_elasticity: float
    churn_impact: ChurnImpact
    margin_optimum_boundary_limited: bool
    margin_optimum_boundary_hit: str
    revenue_optimum: PriceOptimumComparison
    recommendation_basis: str


@dataclass(frozen=True)
class OptimizerData:
    """Loaded datasets and reusable estimates for product-level optimization."""

    product_baselines: dict[str, ProductBaseline]
    category_elasticities: dict[str, float]
    reference_price_change: float
    global_customer_count: int
    global_churn_rate: float


def load_optimizer_data(
    products_path: Path = PRODUCTS_PATH,
    product_events_path: Path = PRODUCT_EVENTS_PATH,
    transactions_path: Path = TRANSACTIONS_PATH,
    customers_path: Path = CUSTOMERS_PATH,
    config: OptimizerConfig = OptimizerConfig(),
) -> OptimizerData:
    """Load raw synthetic datasets and prepare product-level optimizer inputs."""

    products = _load_csv_records(products_path)
    product_events = _load_csv_records(product_events_path)
    transactions = _load_csv_records(transactions_path)
    customers = _load_csv_records(customers_path)

    category_elasticities, product_elasticities = estimate_price_elasticities(
        product_events,
        config=config,
    )
    reference_price_change = estimate_reference_price_change(
        product_events,
        config=config,
    )
    global_customer_count, global_churn_rate = summarize_customer_churn(customers)
    product_baselines = build_product_baselines(
        products=products,
        product_events=product_events,
        transactions=transactions,
        customers=customers,
        product_elasticities=product_elasticities,
        category_elasticities=category_elasticities,
        config=config,
    )

    return OptimizerData(
        product_baselines=product_baselines,
        category_elasticities=category_elasticities,
        reference_price_change=reference_price_change,
        global_customer_count=global_customer_count,
        global_churn_rate=global_churn_rate,
    )


def estimate_price_elasticities(
    product_events: Sequence[dict[str, str]],
    config: OptimizerConfig = OptimizerConfig(),
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate category and product price sensitivity from product-day events.

    Demand is adjusted for the known synthetic seasonality pattern before
    estimating log-log slopes. The fitted slopes are bounded so sparse or noisy
    products cannot produce implausible demand curves.
    """

    category_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    product_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    product_categories: dict[str, str] = {}

    for row in product_events:
        product_id = row.get("product_id", "")
        category = row.get("product_category", "")
        price = _to_float(row.get("selling_price"))
        units = _to_float(row.get("units_sold"))

        product_categories[product_id] = category
        if price <= 0 or _to_bool(row.get("stockout_flag")):
            continue

        seasonality = _seasonality_multiplier_from_iso_date(row.get("event_date", ""))
        seasonally_adjusted_units = max((units + 0.5) / seasonality, 0.01)
        point = (math.log(price), math.log(seasonally_adjusted_units))
        category_points[category].append(point)
        product_points[product_id].append(point)

    category_elasticities: dict[str, float] = {}
    for category, points in category_points.items():
        default = DEFAULT_CATEGORY_ELASTICITIES.get(category, -1.10)
        raw_slope = _ols_slope(points)
        category_elasticities[category] = _bounded_downward_elasticity(
            raw_slope,
            fallback=default,
            config=config,
        )

    product_elasticities: dict[str, float] = {}
    for product_id, points in product_points.items():
        category = product_categories.get(product_id, "")
        category_elasticity = category_elasticities.get(
            category,
            DEFAULT_CATEGORY_ELASTICITIES.get(category, -1.10),
        )

        if len(points) >= config.min_elasticity_rows:
            product_slope = _bounded_downward_elasticity(
                _ols_slope(points),
                fallback=category_elasticity,
                config=config,
            )
        else:
            product_slope = category_elasticity

        blended = (
            config.product_elasticity_weight * product_slope
            + config.category_elasticity_weight * category_elasticity
        )
        product_elasticities[product_id] = _bounded_downward_elasticity(
            blended,
            fallback=category_elasticity,
            config=config,
        )

    return category_elasticities, product_elasticities


def estimate_reference_price_change(
    product_events: Sequence[dict[str, str]],
    config: OptimizerConfig = OptimizerConfig(),
) -> float:
    """Estimate the typical synthetic price increase used in causal analysis."""

    price_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"normal": [], "increase": []}
    )
    for row in product_events:
        product_id = row.get("product_id", "")
        selling_price = _to_float(row.get("selling_price"))
        if selling_price <= 0:
            continue

        key = "increase" if _to_bool(row.get("price_increase_occurred")) else "normal"
        price_groups[product_id][key].append(selling_price)

    observed_lifts: list[float] = []
    for prices in price_groups.values():
        normal_prices = prices["normal"]
        increase_prices = prices["increase"]
        if not normal_prices or not increase_prices:
            continue

        normal_average = statistics.fmean(normal_prices)
        increase_average = statistics.fmean(increase_prices)
        if normal_average > 0 and increase_average > normal_average:
            observed_lifts.append(increase_average / normal_average - 1.0)

    if not observed_lifts:
        return config.default_reference_price_change

    return _clamp(
        statistics.median(observed_lifts),
        0.01,
        0.25,
    )


def summarize_customer_churn(customers: Sequence[dict[str, str]]) -> tuple[int, float]:
    """Return global customer count and churn rate from the customer dataset."""

    customer_ids = {
        row.get("customer_id", "")
        for row in customers
        if row.get("customer_id", "")
    }
    churned_customers = {
        row.get("customer_id", "")
        for row in customers
        if row.get("customer_id", "") and _to_bool(row.get("churned"))
    }
    customer_count = len(customer_ids)
    churn_rate = _safe_divide(len(churned_customers), customer_count)
    return customer_count, churn_rate


def build_product_baselines(
    products: Sequence[dict[str, str]],
    product_events: Sequence[dict[str, str]],
    transactions: Sequence[dict[str, str]],
    customers: Sequence[dict[str, str]],
    product_elasticities: dict[str, float],
    category_elasticities: dict[str, float],
    config: OptimizerConfig = OptimizerConfig(),
) -> dict[str, ProductBaseline]:
    """Build observed product baselines from raw product and event data."""

    events_by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in product_events:
        events_by_product[row.get("product_id", "")].append(row)

    product_customer_ids: dict[str, set[str]] = defaultdict(set)
    for row in transactions:
        product_id = row.get("product_id", "")
        customer_id = row.get("customer_id", "")
        if product_id and customer_id:
            product_customer_ids[product_id].add(customer_id)

    customer_churn_lookup = {
        row.get("customer_id", ""): _to_bool(row.get("churned"))
        for row in customers
        if row.get("customer_id", "")
    }

    baselines: dict[str, ProductBaseline] = {}
    for product in products:
        product_id = product.get("product_id", "")
        product_name = product.get("product_name", "")
        category = product.get("product_category", "")
        unit_cost = _to_float(product.get("unit_cost"))
        product_rows = events_by_product.get(product_id, [])

        observed_days = len(product_rows)
        baseline_units = sum(_to_float(row.get("units_sold")) for row in product_rows)
        baseline_revenue = sum(_to_float(row.get("revenue")) for row in product_rows)
        baseline_margin = sum(_to_float(row.get("gross_margin")) for row in product_rows)
        product_views = int(sum(_to_float(row.get("product_views")) for row in product_rows))
        purchases = int(sum(_to_float(row.get("purchases")) for row in product_rows))
        stockout_days = sum(1 for row in product_rows if _to_bool(row.get("stockout_flag")))
        price_increase_days = sum(
            1 for row in product_rows if _to_bool(row.get("price_increase_occurred"))
        )

        current_average_selling_price = _safe_divide(
            baseline_revenue,
            baseline_units,
            default=_to_float(product.get("total_revenue"))
            / max(_to_float(product.get("total_units_sold")), 1.0),
        )
        if current_average_selling_price <= 0:
            current_average_selling_price = _to_float(product.get("list_price"))

        customer_ids = product_customer_ids.get(product_id, set())
        churned_product_customers = sum(
            1 for customer_id in customer_ids if customer_churn_lookup.get(customer_id, False)
        )
        product_churn_rate = _safe_divide(churned_product_customers, len(customer_ids))
        category_elasticity = category_elasticities.get(
            category,
            DEFAULT_CATEGORY_ELASTICITIES.get(category, -1.10),
        )
        product_elasticity = product_elasticities.get(product_id, category_elasticity)

        min_price = max(
            current_average_selling_price * config.min_price_multiplier,
            unit_cost * config.min_margin_price_cost_multiple,
        )
        max_price = max(current_average_selling_price * config.max_price_multiplier, min_price)

        baselines[product_id] = ProductBaseline(
            product_id=product_id,
            product_name=product_name,
            product_category=category,
            unit_cost=unit_cost,
            current_average_selling_price=current_average_selling_price,
            baseline_unit_velocity=_safe_divide(baseline_units, observed_days),
            baseline_units_sold=baseline_units,
            baseline_revenue=baseline_revenue,
            baseline_gross_margin=baseline_margin,
            baseline_gross_margin_rate=_safe_divide(baseline_margin, baseline_revenue),
            observed_product_days=observed_days,
            product_views=product_views,
            purchases=purchases,
            stockout_day_rate=_safe_divide(stockout_days, observed_days),
            price_increase_day_rate=_safe_divide(price_increase_days, observed_days),
            unique_customers=len(customer_ids),
            baseline_product_churn_rate=product_churn_rate,
            estimated_price_elasticity=product_elasticity,
            category_price_elasticity=category_elasticity,
            min_tested_price=round(min_price, 2),
            max_tested_price=round(max_price, 2),
        )

    return baselines


def evaluate_product_price_scenario(
    product_id: str,
    proposed_price: float,
    data: OptimizerData,
    scenario_label: str = "Proposed price",
    config: OptimizerConfig = OptimizerConfig(),
) -> ProductPriceScenarioResult:
    """Evaluate revenue, margin, demand, and churn for one product price."""

    if product_id not in data.product_baselines:
        available = ", ".join(sorted(data.product_baselines)[:5])
        raise ValueError(f"Unknown product_id '{product_id}'. Examples: {available}")

    baseline = data.product_baselines[product_id]
    _validate_proposed_price(proposed_price, baseline)

    price_change_pct = proposed_price / baseline.current_average_selling_price - 1.0

    demand_multiplier = estimate_local_demand_multiplier(
        price_change_pct=price_change_pct,
        estimated_elasticity=baseline.estimated_price_elasticity,
        reference_price_change=data.reference_price_change,
        config=config,
    )

    churn_impact = estimate_customer_churn_impact(
        price_change_pct=price_change_pct,
        baseline=baseline,
        data=data,
        config=config,
    )
    expected_unit_velocity = (
        baseline.baseline_unit_velocity
        * demand_multiplier
        * churn_impact.retention_demand_multiplier
    )
    expected_units = expected_unit_velocity * baseline.observed_product_days
    unit_margin = max(proposed_price - baseline.unit_cost, 0.0)
    expected_revenue = expected_units * proposed_price
    expected_margin = expected_units * unit_margin

    return ProductPriceScenarioResult(
        product_id=baseline.product_id,
        product_name=baseline.product_name,
        product_category=baseline.product_category,
        scenario_label=scenario_label,
        proposed_price=round(proposed_price, 2),
        current_average_selling_price=baseline.current_average_selling_price,
        price_change_pct=price_change_pct,
        estimated_price_elasticity=baseline.estimated_price_elasticity,
        baseline_unit_velocity=baseline.baseline_unit_velocity,
        expected_unit_velocity=expected_unit_velocity,
        expected_units_sold=expected_units,
        expected_revenue=expected_revenue,
        expected_gross_margin=expected_margin,
        revenue_change_vs_baseline=expected_revenue - baseline.baseline_revenue,
        margin_change_vs_baseline=expected_margin - baseline.baseline_gross_margin,
        demand_multiplier=demand_multiplier,
        churn_impact=churn_impact,
    )


def estimate_local_demand_multiplier(
    price_change_pct: float,
    estimated_elasticity: float,
    reference_price_change: float,
    config: OptimizerConfig = OptimizerConfig(),
) -> float:
    """Estimate demand response while treating elasticity as a local estimate.

    The observed synthetic price variation is modest, so the model applies the
    fitted elasticity at full strength near the observed reference lift and then
    gradually softens additional extrapolation. This preserves a smooth,
    downward-sloping demand curve without over-trusting large unobserved price
    moves.
    """

    if price_change_pct <= -1.0:
        raise ValueError("price_change_pct must be greater than -100%.")

    log_price_change = math.log1p(price_change_pct)
    adjusted_log_price_change = _soften_extrapolated_log_price_change(
        log_price_change=log_price_change,
        reference_price_change=reference_price_change,
        config=config,
    )
    raw_demand_multiplier = math.exp(estimated_elasticity * adjusted_log_price_change)
    return _clamp(
        raw_demand_multiplier,
        config.demand_floor_multiplier,
        config.demand_ceiling_multiplier,
    )


def estimate_customer_churn_impact(
    price_change_pct: float,
    baseline: ProductBaseline,
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> ChurnImpact:
    """Estimate secondary retention impact from the causal churn estimate.

    The causal estimate is treated as a local customer-retention adjustment, not
    as the pricing engine. The product demand curve remains driven by product
    and category elasticity estimated from product-event data.
    """

    baseline_churn_rate = (
        baseline.baseline_product_churn_rate
        if baseline.unique_customers > 0
        else data.global_churn_rate
    )
    if data.reference_price_change <= 0:
        reference_change = config.default_reference_price_change
    else:
        reference_change = data.reference_price_change

    churn_delta = config.causal_churn_effect * (price_change_pct / reference_change)
    if price_change_pct < 0:
        churn_delta *= config.price_decrease_churn_relief_factor

    churn_delta = _clamp(
        churn_delta,
        -config.max_abs_churn_rate_change,
        config.max_abs_churn_rate_change,
    )
    estimated_churn_rate = _clamp(baseline_churn_rate + churn_delta, 0.0, 1.0)

    baseline_retention_rate = max(1.0 - baseline_churn_rate, 0.01)
    estimated_retention_rate = max(1.0 - estimated_churn_rate, 0.0)
    retention_multiplier = _clamp(
        estimated_retention_rate / baseline_retention_rate,
        config.min_retention_adjustment_multiplier,
        config.max_retention_adjustment_multiplier,
    )

    exposed_customers = baseline.unique_customers or data.global_customer_count
    incremental_churned_customers = churn_delta * exposed_customers
    basis = (
        "Secondary adjustment from notebook 04 adjusted IPW estimate "
        f"({config.causal_churn_effect * 100:.2f} pp churn change at an observed "
        f"{reference_change * 100:.1f}% reference price lift); predictive churn model not used."
    )

    return ChurnImpact(
        baseline_churn_rate=baseline_churn_rate,
        estimated_churn_rate=estimated_churn_rate,
        churn_rate_change=estimated_churn_rate - baseline_churn_rate,
        estimated_incremental_churned_customers=incremental_churned_customers,
        retention_demand_multiplier=retention_multiplier,
        causal_effect_basis=basis,
    )


def build_product_price_grid(
    baseline: ProductBaseline,
    config: OptimizerConfig = OptimizerConfig(),
) -> list[float]:
    """Build a bounded price grid for product-level scenario testing."""

    prices = {
        round(baseline.current_average_selling_price, 2),
        round(baseline.min_tested_price, 2),
        round(baseline.max_tested_price, 2),
    }
    multiplier = config.min_price_multiplier
    while multiplier <= config.max_price_multiplier + 1e-9:
        price = round(baseline.current_average_selling_price * multiplier, 2)
        if baseline.min_tested_price <= price <= baseline.max_tested_price:
            prices.add(price)
        multiplier += config.price_grid_step_pct

    return sorted(prices)


def evaluate_product_price_scenarios(
    product_id: str,
    proposed_prices: Iterable[float],
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> list[ProductPriceScenarioResult]:
    """Evaluate several candidate prices for a selected product."""

    return [
        evaluate_product_price_scenario(
            product_id=product_id,
            proposed_price=price,
            data=data,
            scenario_label=f"${price:,.2f}",
            config=config,
        )
        for price in proposed_prices
    ]


def find_revenue_maximizing_price(
    product_id: str,
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> ProductPriceScenarioResult:
    """Find the tested price that maximizes expected product revenue."""

    return find_product_optimal_prices(
        product_id=product_id,
        data=data,
        config=config,
    ).revenue_optimal_scenario


def find_margin_maximizing_price(
    product_id: str,
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> ProductPriceScenarioResult:
    """Find the tested price that maximizes expected product gross margin."""

    return find_product_optimal_prices(
        product_id=product_id,
        data=data,
        config=config,
    ).margin_optimal_scenario


def find_product_optimal_prices(
    product_id: str,
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> ProductOptimizationResult:
    """Find revenue and gross-margin optima within the tested local price range."""

    baseline = data.product_baselines[product_id]
    price_grid = build_product_price_grid(baseline, config=config)
    scenarios = evaluate_product_price_scenarios(
        product_id=product_id,
        proposed_prices=price_grid,
        data=data,
        config=config,
    )
    revenue_optimal = max(scenarios, key=lambda scenario: scenario.expected_revenue)
    margin_optimal = max(scenarios, key=lambda scenario: scenario.expected_gross_margin)

    return ProductOptimizationResult(
        product_id=baseline.product_id,
        product_name=baseline.product_name,
        product_category=baseline.product_category,
        revenue_optimal_scenario=revenue_optimal,
        margin_optimal_scenario=margin_optimal,
        revenue_optimum_boundary_limited=_is_boundary_price(
            revenue_optimal.proposed_price,
            baseline,
        ),
        margin_optimum_boundary_limited=_is_boundary_price(
            margin_optimal.proposed_price,
            baseline,
        ),
        min_tested_price=baseline.min_tested_price,
        max_tested_price=baseline.max_tested_price,
        tested_price_count=len(price_grid),
    )


def build_product_price_recommendation(
    product_id: str,
    data: OptimizerData,
    config: OptimizerConfig = OptimizerConfig(),
) -> ProductPriceRecommendation:
    """Build the dashboard-facing product price recommendation.

    Gross margin is the primary recommendation target because revenue-only
    optimization can favor aggressive discounts when demand is elastic. The
    revenue optimum is still returned as a secondary comparison for context.
    """

    baseline = data.product_baselines[product_id]
    optima = find_product_optimal_prices(
        product_id=product_id,
        data=data,
        config=config,
    )
    recommended = optima.margin_optimal_scenario
    revenue_optimum = optima.revenue_optimal_scenario

    return ProductPriceRecommendation(
        product_id=baseline.product_id,
        product_name=baseline.product_name,
        product_category=baseline.product_category,
        current_price=baseline.current_average_selling_price,
        recommended_price=recommended.proposed_price,
        recommended_percent_change=recommended.price_change_pct,
        expected_units=recommended.expected_units_sold,
        expected_revenue=recommended.expected_revenue,
        expected_gross_margin=recommended.expected_gross_margin,
        revenue_delta=recommended.revenue_change_vs_baseline,
        margin_delta=recommended.margin_change_vs_baseline,
        estimated_elasticity=baseline.estimated_price_elasticity,
        churn_impact=recommended.churn_impact,
        margin_optimum_boundary_limited=optima.margin_optimum_boundary_limited,
        margin_optimum_boundary_hit=_boundary_side(
            recommended.proposed_price,
            baseline,
        ),
        revenue_optimum=_build_optimum_comparison(
            scenario=revenue_optimum,
            boundary_limited=optima.revenue_optimum_boundary_limited,
            boundary_hit=_boundary_side(revenue_optimum.proposed_price, baseline),
        ),
        recommendation_basis=(
            "Recommended price is the gross-margin-maximizing tested price. "
            "Revenue-optimal pricing is retained as a secondary comparison "
            "because elastic demand can make revenue maximization favor "
            "aggressive discounts that weaken gross margin."
        ),
    )


def product_recommendation_to_dict(
    recommendation: ProductPriceRecommendation,
) -> dict[str, Any]:
    """Convert a recommendation into a plain dictionary for API/dashboard use."""

    return {
        "product_id": recommendation.product_id,
        "product_name": recommendation.product_name,
        "product_category": recommendation.product_category,
        "current_price": recommendation.current_price,
        "recommended_price": recommendation.recommended_price,
        "recommended_percent_change": recommendation.recommended_percent_change,
        "expected_units": recommendation.expected_units,
        "expected_revenue": recommendation.expected_revenue,
        "expected_gross_margin": recommendation.expected_gross_margin,
        "revenue_delta": recommendation.revenue_delta,
        "margin_delta": recommendation.margin_delta,
        "estimated_elasticity": recommendation.estimated_elasticity,
        "baseline_churn_rate": recommendation.churn_impact.baseline_churn_rate,
        "estimated_churn_rate": recommendation.churn_impact.estimated_churn_rate,
        "churn_rate_change": recommendation.churn_impact.churn_rate_change,
        "estimated_incremental_churned_customers": (
            recommendation.churn_impact.estimated_incremental_churned_customers
        ),
        "retention_demand_multiplier": (
            recommendation.churn_impact.retention_demand_multiplier
        ),
        "causal_effect_basis": recommendation.churn_impact.causal_effect_basis,
        "margin_optimum_boundary_limited": (
            recommendation.margin_optimum_boundary_limited
        ),
        "margin_optimum_boundary_hit": recommendation.margin_optimum_boundary_hit,
        "revenue_optimum": {
            "price": recommendation.revenue_optimum.price,
            "percent_change": recommendation.revenue_optimum.percent_change,
            "expected_units": recommendation.revenue_optimum.expected_units,
            "expected_revenue": recommendation.revenue_optimum.expected_revenue,
            "expected_gross_margin": (
                recommendation.revenue_optimum.expected_gross_margin
            ),
            "revenue_delta": recommendation.revenue_optimum.revenue_delta,
            "margin_delta": recommendation.revenue_optimum.margin_delta,
            "boundary_limited": recommendation.revenue_optimum.boundary_limited,
            "boundary_hit": recommendation.revenue_optimum.boundary_hit,
        },
        "recommendation_basis": recommendation.recommendation_basis,
    }


def scenario_results_to_dicts(
    scenarios: Sequence[ProductPriceScenarioResult],
) -> list[dict[str, Any]]:
    """Convert scenario dataclasses into plain dictionaries for dashboards."""

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "product_id": scenario.product_id,
                "product_name": scenario.product_name,
                "product_category": scenario.product_category,
                "scenario_label": scenario.scenario_label,
                "proposed_price": scenario.proposed_price,
                "current_average_selling_price": scenario.current_average_selling_price,
                "price_change_pct": scenario.price_change_pct,
                "estimated_price_elasticity": scenario.estimated_price_elasticity,
                "baseline_unit_velocity": scenario.baseline_unit_velocity,
                "expected_unit_velocity": scenario.expected_unit_velocity,
                "expected_units_sold": scenario.expected_units_sold,
                "expected_revenue": scenario.expected_revenue,
                "expected_gross_margin": scenario.expected_gross_margin,
                "revenue_change_vs_baseline": scenario.revenue_change_vs_baseline,
                "margin_change_vs_baseline": scenario.margin_change_vs_baseline,
                "demand_multiplier": scenario.demand_multiplier,
                "baseline_churn_rate": scenario.churn_impact.baseline_churn_rate,
                "estimated_churn_rate": scenario.churn_impact.estimated_churn_rate,
                "churn_rate_change": scenario.churn_impact.churn_rate_change,
                "estimated_incremental_churned_customers": (
                    scenario.churn_impact.estimated_incremental_churned_customers
                ),
                "retention_demand_multiplier": (
                    scenario.churn_impact.retention_demand_multiplier
                ),
                "causal_effect_basis": scenario.churn_impact.causal_effect_basis,
            }
        )
    return rows


def select_representative_products(data: OptimizerData) -> list[str]:
    """Select top, middle, and slower products for the manual terminal test."""

    baselines = list(data.product_baselines.values())
    if not baselines:
        return []

    by_revenue = sorted(baselines, key=lambda item: item.baseline_revenue, reverse=True)
    by_velocity = sorted(baselines, key=lambda item: item.baseline_unit_velocity)
    selected = [by_revenue[0].product_id]

    median_revenue_product = by_revenue[len(by_revenue) // 2].product_id
    if median_revenue_product not in selected:
        selected.append(median_revenue_product)

    for baseline in by_velocity:
        if baseline.product_id not in selected:
            selected.append(baseline.product_id)
            break

    return selected[:3]


def print_manual_product_tests(config: OptimizerConfig = OptimizerConfig()) -> None:
    """Print representative product pricing scenarios for quick validation."""

    data = load_optimizer_data(config=config)
    product_ids = select_representative_products(data)
    print("Product-Level Revenue Optimizer Manual Test")
    print(f"Products loaded: {len(data.product_baselines):,}")
    print(f"Observed reference price lift: {data.reference_price_change * 100:.1f}%")
    print(
        "Recommended price uses gross-margin maximization; revenue-only "
        "optimization is shown as a secondary comparison because elastic demand "
        "can favor aggressive discounts."
    )
    print()

    for product_id in product_ids:
        baseline = data.product_baselines[product_id]
        optimal_prices = find_product_optimal_prices(product_id, data, config=config)
        recommendation = build_product_price_recommendation(
            product_id=product_id,
            data=data,
            config=config,
        )
        scenario_inputs = [
            ("Current price", baseline.current_average_selling_price),
            ("-10%", baseline.current_average_selling_price * 0.90),
            ("+10%", baseline.current_average_selling_price * 1.10),
            (
                "Revenue-optimal",
                optimal_prices.revenue_optimal_scenario.proposed_price,
            ),
            (
                "Recommended price",
                recommendation.recommended_price,
            ),
        ]
        scenarios = [
            evaluate_product_price_scenario(
                product_id=product_id,
                proposed_price=price,
                data=data,
                scenario_label=label,
                config=config,
            )
            for label, price in scenario_inputs
        ]

        print(
            f"{baseline.product_name} ({baseline.product_id}, "
            f"{baseline.product_category})"
        )
        print(
            "  Baseline: "
            f"avg price {_format_currency(baseline.current_average_selling_price)}, "
            f"unit velocity {baseline.baseline_unit_velocity:.2f}/day, "
            f"elasticity {baseline.estimated_price_elasticity:.2f}, "
            f"revenue {_format_currency(baseline.baseline_revenue)}, "
            f"margin {_format_currency(baseline.baseline_gross_margin)}, "
            f"buyer churn {_format_percent(baseline.baseline_product_churn_rate)}"
        )
        print(
            "  Tested range: "
            f"{_format_currency(optimal_prices.min_tested_price)} to "
            f"{_format_currency(optimal_prices.max_tested_price)} "
            f"({optimal_prices.tested_price_count} prices)"
        )
        print(
            "  Boundary-limited optima: "
            f"revenue={_format_bool(optimal_prices.revenue_optimum_boundary_limited)}, "
            f"margin={_format_bool(optimal_prices.margin_optimum_boundary_limited)}"
        )
        print(
            "  Recommended price: "
            f"{_format_currency(recommendation.recommended_price)} "
            f"({_format_percent(recommendation.recommended_percent_change, signed=True)}, "
            "gross-margin optimal)"
        )
        _print_scenario_table(
            scenarios,
            boundary_status={
                "Revenue-optimal": optimal_prices.revenue_optimum_boundary_limited,
                "Recommended price": recommendation.margin_optimum_boundary_limited,
            },
        )
        print()


def print_optimizer_diagnostics(config: OptimizerConfig = OptimizerConfig()) -> None:
    """Print all-product optimizer diagnostics without changing model behavior."""

    data = load_optimizer_data(config=config)
    product_rows = []

    for baseline in sorted(
        data.product_baselines.values(),
        key=lambda item: (item.product_category, item.product_name),
    ):
        optima = find_product_optimal_prices(
            product_id=baseline.product_id,
            data=data,
            config=config,
        )
        revenue_side = _boundary_side(
            optima.revenue_optimal_scenario.proposed_price,
            baseline,
        )
        margin_side = _boundary_side(
            optima.margin_optimal_scenario.proposed_price,
            baseline,
        )
        product_rows.append((baseline, optima, revenue_side, margin_side))

    print("Product-Level Optimizer Diagnostics")
    print(f"Products evaluated: {len(product_rows):,}")
    print(f"Default search range: {config.min_price_multiplier - 1:+.0%} to "
          f"{config.max_price_multiplier - 1:+.0%}")
    print(f"Observed reference price lift: {data.reference_price_change * 100:.1f}%")
    print()
    _print_diagnostic_product_table(product_rows)
    print()
    _print_aggregate_diagnostics(product_rows)


def _print_diagnostic_product_table(
    product_rows: Sequence[
        tuple[ProductBaseline, ProductOptimizationResult, str, str]
    ],
) -> None:
    """Print concise per-product optimizer diagnostics."""

    headers = [
        "Product",
        "Category",
        "Elasticity",
        "Current",
        "Rev Opt",
        "Recommended",
        "Rev Bound",
        "Rev Hit",
        "Rec Bound",
        "Rec Hit",
    ]
    widths = [26, 14, 10, 9, 9, 11, 9, 8, 9, 10]
    print("  " + " ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  " + " ".join("-" * width for width in widths))

    for baseline, optima, revenue_side, margin_side in product_rows:
        revenue_optimum = optima.revenue_optimal_scenario
        margin_optimum = optima.margin_optimal_scenario
        row = [
            _truncate_text(baseline.product_name, widths[0]),
            baseline.product_category,
            f"{baseline.estimated_price_elasticity:.2f}",
            _format_currency(baseline.current_average_selling_price),
            _format_percent(revenue_optimum.price_change_pct, signed=True),
            _format_percent(margin_optimum.price_change_pct, signed=True),
            _format_bool(optima.revenue_optimum_boundary_limited),
            _format_boundary_hit(revenue_side, baseline),
            _format_bool(optima.margin_optimum_boundary_limited),
            _format_boundary_hit(margin_side, baseline),
        ]
        print("  " + " ".join(value.ljust(width) for value, width in zip(row, widths)))


def _print_aggregate_diagnostics(
    product_rows: Sequence[
        tuple[ProductBaseline, ProductOptimizationResult, str, str]
    ],
) -> None:
    """Print aggregate diagnostics for optimizer shape and boundary behavior."""

    elasticities = [row[0].estimated_price_elasticity for row in product_rows]
    revenue_lower = sum(1 for _, _, revenue_side, _ in product_rows if revenue_side == "lower")
    revenue_upper = sum(1 for _, _, revenue_side, _ in product_rows if revenue_side == "upper")
    margin_lower = sum(1 for _, _, _, margin_side in product_rows if margin_side == "lower")
    margin_upper = sum(1 for _, _, _, margin_side in product_rows if margin_side == "upper")
    revenue_interior = sum(
        1 for _, _, revenue_side, _ in product_rows if revenue_side == "interior"
    )
    margin_interior = sum(
        1 for _, _, _, margin_side in product_rows if margin_side == "interior"
    )
    any_interior = sum(
        1
        for _, _, revenue_side, margin_side in product_rows
        if "interior" in {revenue_side, margin_side}
    )
    both_interior = sum(
        1
        for _, _, revenue_side, margin_side in product_rows
        if revenue_side == "interior" and margin_side == "interior"
    )

    print("Aggregate Diagnostics")
    print(
        "  Elasticity estimates: "
        f"median {statistics.median(elasticities):.2f}, "
        f"range {min(elasticities):.2f} to {max(elasticities):.2f}"
    )
    print(f"  Revenue optima at -15%: {revenue_lower}")
    print(f"  Revenue optima at +15%: {revenue_upper}")
    print(f"  Margin optima at -15%: {margin_lower}")
    print(f"  Margin optima at +15%: {margin_upper}")
    print(f"  Products with an interior optimum: {any_interior}")
    print(f"  Interior revenue optima: {revenue_interior}")
    print(f"  Interior margin optima: {margin_interior}")
    print(f"  Products with both optima interior: {both_interior}")


def _build_optimum_comparison(
    scenario: ProductPriceScenarioResult,
    boundary_limited: bool,
    boundary_hit: str,
) -> PriceOptimumComparison:
    """Summarize a tested optimum for comparison in recommendation outputs."""

    return PriceOptimumComparison(
        price=scenario.proposed_price,
        percent_change=scenario.price_change_pct,
        expected_units=scenario.expected_units_sold,
        expected_revenue=scenario.expected_revenue,
        expected_gross_margin=scenario.expected_gross_margin,
        revenue_delta=scenario.revenue_change_vs_baseline,
        margin_delta=scenario.margin_change_vs_baseline,
        boundary_limited=boundary_limited,
        boundary_hit=boundary_hit,
    )


def _print_scenario_table(
    scenarios: Sequence[ProductPriceScenarioResult],
    boundary_status: dict[str, bool] | None = None,
) -> None:
    """Print a compact fixed-width table for terminal scenario checks."""

    boundary_status = boundary_status or {}
    headers = [
        "Scenario",
        "Price",
        "Change",
        "Units",
        "Velocity",
        "Churn",
        "Revenue",
        "Margin",
        "Rev Delta",
        "Margin Delta",
        "Boundary",
    ]
    widths = [19, 10, 9, 10, 10, 9, 13, 13, 13, 14, 10]
    print("  " + " ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  " + " ".join("-" * width for width in widths))
    for scenario in scenarios:
        row = [
            scenario.scenario_label,
            _format_currency(scenario.proposed_price),
            _format_percent(scenario.price_change_pct, signed=True),
            f"{scenario.expected_units_sold:,.0f}",
            f"{scenario.expected_unit_velocity:.2f}/day",
            _format_percent(scenario.churn_impact.estimated_churn_rate),
            _format_currency(scenario.expected_revenue),
            _format_currency(scenario.expected_gross_margin),
            _format_signed_currency(scenario.revenue_change_vs_baseline),
            _format_signed_currency(scenario.margin_change_vs_baseline),
            "yes" if boundary_status.get(scenario.scenario_label, False) else "",
        ]
        print("  " + " ".join(value.ljust(width) for value, width in zip(row, widths)))


def _validate_proposed_price(proposed_price: float, baseline: ProductBaseline) -> None:
    """Raise a clear error when a tested product price is outside bounds."""

    if not math.isfinite(proposed_price):
        raise ValueError("proposed_price must be finite.")
    tolerance = 0.005
    if (
        proposed_price < baseline.min_tested_price - tolerance
        or proposed_price > baseline.max_tested_price + tolerance
    ):
        raise ValueError(
            f"proposed_price for {baseline.product_id} must be between "
            f"${baseline.min_tested_price:.2f} and ${baseline.max_tested_price:.2f}."
        )


def _load_csv_records(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into dictionaries while validating that it exists."""

    if not path.exists():
        raise FileNotFoundError(f"Required optimizer input not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _seasonality_multiplier_from_iso_date(date_text: str) -> float:
    """Approximate the synthetic monthly seasonality used during generation."""

    try:
        month = int(date_text[5:7])
    except (TypeError, ValueError):
        return 1.0

    if month in {11, 12}:
        return 1.45
    if month in {1, 6, 7}:
        return 1.12
    if month in {2, 8}:
        return 0.92
    return 1.0


def _ols_slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Return the ordinary least squares slope for ``(x, y)`` points."""

    if len(points) < 2:
        return None

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator <= 0:
        return None

    return numerator / denominator


def _soften_extrapolated_log_price_change(
    log_price_change: float,
    reference_price_change: float,
    config: OptimizerConfig,
) -> float:
    """Preserve local elasticity while softening movement beyond observed support."""

    observed_support = _clamp(
        abs(reference_price_change),
        config.min_local_response_change,
        config.max_local_response_change,
    )
    local_log_support = math.log1p(observed_support)
    absolute_log_change = abs(log_price_change)
    if absolute_log_change <= local_log_support:
        return log_price_change

    extra_log_change = absolute_log_change - local_log_support
    softened_extra = (
        config.extrapolated_response_weight * extra_log_change
        + (1.0 - config.extrapolated_response_weight)
        * local_log_support
        * (1.0 - math.exp(-extra_log_change / local_log_support))
    )
    adjusted_log_change = local_log_support + softened_extra
    return math.copysign(adjusted_log_change, log_price_change)


def _is_boundary_price(price: float, baseline: ProductBaseline) -> bool:
    """Return whether a tested optimum sits on the local search boundary."""

    tolerance = 0.005
    return (
        abs(price - baseline.min_tested_price) <= tolerance
        or abs(price - baseline.max_tested_price) <= tolerance
    )


def _boundary_side(price: float, baseline: ProductBaseline) -> str:
    """Identify whether a price is on the lower boundary, upper boundary, or inside."""

    tolerance = 0.005
    if abs(price - baseline.min_tested_price) <= tolerance:
        return "lower"
    if abs(price - baseline.max_tested_price) <= tolerance:
        return "upper"
    return "interior"


def _format_boundary_hit(side: str, baseline: ProductBaseline) -> str:
    """Format which tested search boundary was hit."""

    if side == "lower":
        boundary_pct = (
            baseline.min_tested_price / baseline.current_average_selling_price - 1.0
        )
        return _format_compact_percent(boundary_pct, signed=True)
    if side == "upper":
        boundary_pct = (
            baseline.max_tested_price / baseline.current_average_selling_price - 1.0
        )
        return _format_compact_percent(boundary_pct, signed=True)
    return "interior"


def _bounded_downward_elasticity(
    value: float | None,
    fallback: float,
    config: OptimizerConfig,
) -> float:
    """Keep elasticity negative and within realistic synthetic-product bounds."""

    if value is None or not math.isfinite(value) or value >= 0:
        value = fallback
    return _clamp(value, config.min_elasticity, config.max_elasticity)


def _to_bool(value: Any) -> bool:
    """Convert CSV truthy values into booleans."""

    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert CSV numeric values into floats with a safe default."""

    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numbers while avoiding undefined zero-denominator results."""

    if denominator == 0:
        return default
    return numerator / denominator


def _clamp(value: float, lower: float, upper: float) -> float:
    """Bound a value within the closed interval ``[lower, upper]``."""

    return max(lower, min(upper, value))


def _format_currency(value: float) -> str:
    """Format a float as a dollar value."""

    return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"


def _format_signed_currency(value: float) -> str:
    """Format a float as a signed dollar value."""

    if abs(value) < 0.005:
        value = 0.0
    sign = "+" if value >= 0 else "-"
    absolute_value = abs(value)
    if absolute_value >= 1000:
        return f"{sign}${absolute_value:,.0f}"
    return f"{sign}${absolute_value:,.2f}"


def _format_percent(value: float, signed: bool = False) -> str:
    """Format a decimal value as a percentage."""

    if abs(value) < 0.0005:
        value = 0.0
    if signed:
        return f"{value:+.1%}"
    return f"{value:.1%}"


def _format_compact_percent(value: float, signed: bool = False) -> str:
    """Format whole-number percentages without trailing decimals."""

    return _format_percent(value, signed=signed).replace(".0%", "%")


def _format_bool(value: bool) -> str:
    """Format booleans for the manual terminal table."""

    return "yes" if value else "no"


def _truncate_text(value: str, width: int) -> str:
    """Fit text into a fixed-width terminal column."""

    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def main(argv: Sequence[str] | None = None) -> None:
    """Run either the representative manual test or full diagnostics."""

    parser = argparse.ArgumentParser(
        description="Product-level revenue optimizer for synthetic ecommerce data."
    )
    parser.add_argument(
        "--diagnostics",
        "--diagnostic",
        action="store_true",
        help="Print optimizer diagnostics for every product.",
    )
    args = parser.parse_args(argv)

    if args.diagnostics:
        print_optimizer_diagnostics()
    else:
        print_manual_product_tests()


if __name__ == "__main__":
    main()
