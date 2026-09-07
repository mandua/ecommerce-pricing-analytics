"""FastAPI backend for the local ecommerce analytics dashboard."""

from __future__ import annotations

import math
import sys
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revenue_optimizer import (  # noqa: E402
    OptimizerConfig,
    build_product_price_grid,
    build_product_price_recommendation,
    evaluate_product_price_scenario,
    evaluate_product_price_scenarios,
    find_product_optimal_prices,
    load_optimizer_data,
    product_recommendation_to_dict,
    scenario_results_to_dicts,
)


app = FastAPI(
    title="PriceLabs",
    description="Local API for synthetic/model-based ecommerce pricing analytics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_csv_data() -> dict[str, list[dict[str, str]]]:
    """Load generated raw datasets once for local dashboard requests."""

    return {
        "customers": _load_csv(RAW_DATA_DIR / "customers.csv"),
        "transactions": _load_csv(RAW_DATA_DIR / "transactions.csv"),
        "products": _load_csv(RAW_DATA_DIR / "products.csv"),
        "product_events": _load_csv(RAW_DATA_DIR / "product_events.csv"),
    }


@lru_cache(maxsize=1)
def load_pricing_data():
    """Load reusable product-level optimizer state."""

    return load_optimizer_data(config=OptimizerConfig())


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a simple local health check."""

    return {"status": "ok"}


@app.get("/api/metadata")
def get_metadata() -> dict[str, object]:
    """Return filter options and project-level dashboard labels."""

    data = load_csv_data()
    customers = data["customers"]
    products = data["products"]
    date_range = _available_date_bounds(data)

    return {
        "project_name": "PriceLabs",
        "disclaimer": (
            "PriceLabs uses synthetic Gymshark-style ecommerce data for portfolio "
            "analytics only; it is not affiliated with Gymshark and is not a real "
            "Gymshark pricing recommendation."
        ),
        "date_range": date_range,
        "regions": sorted({row["customer_region"] for row in customers}),
        "categories": sorted({row["product_category"] for row in products}),
        "product_count": len(products),
    }


@app.get("/api/executive")
def get_executive_dashboard(
    start_date: str | None = None,
    end_date: str | None = None,
    region: str = "all",
    category: str = "all",
) -> dict[str, object]:
    """Return filtered executive ecommerce metrics."""

    data = load_csv_data()
    date_range = _available_date_bounds(data)
    start_date, end_date = _clamp_date_range(start_date, end_date, date_range)
    transactions = _filter_transactions(
        data["transactions"],
        start_date=start_date,
        end_date=end_date,
        region=region,
        category=category,
    )
    product_events = _filter_product_events(
        data["product_events"],
        start_date=start_date,
        end_date=end_date,
        category=category,
    )
    customers = _customers_for_filtered_context(
        customers=data["customers"],
        transactions=transactions,
        region=region,
    )

    return {
        "kpis": _build_executive_kpis(transactions, customers),
        "funnel": _build_funnel_summary(product_events, region=region),
        "monthly_trends": _build_monthly_trends(transactions),
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "region": region,
            "category": category,
        },
    }


@app.get("/api/product-analytics")
def get_product_analytics(
    product_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    region: str = "all",
    category: str = "all",
) -> dict[str, object]:
    """Return product-level ecommerce analytics."""

    raw_data = load_csv_data()
    date_range = _available_date_bounds(raw_data)
    start_date, end_date = _clamp_date_range(start_date, end_date, date_range)
    optimizer_data = load_pricing_data()
    transactions = _filter_transactions(
        raw_data["transactions"],
        start_date=start_date,
        end_date=end_date,
        region=region,
        category=category,
    )
    product_events = _filter_product_events(
        raw_data["product_events"],
        start_date=start_date,
        end_date=end_date,
        category=category,
    )
    product_summaries = _build_product_summaries(
        product_events=product_events,
        transactions=transactions,
        optimizer_data=optimizer_data,
    )
    if not product_summaries:
        return {
            "products": [],
            "selected_product": None,
            "top_bottom": {},
            "category_performance": [],
            "note": (
                "No product transactions matched the selected filters. "
                "Product-day inventory and view data is not region-specific."
            ),
        }

    selected_product_id = product_id or product_summaries[0]["product_id"]
    if selected_product_id not in {product["product_id"] for product in product_summaries}:
        selected_product_id = product_summaries[0]["product_id"]

    selected_product = next(
        product for product in product_summaries if product["product_id"] == selected_product_id
    )

    return {
        "products": product_summaries,
        "selected_product": selected_product,
        "top_bottom": _build_top_bottom_products(product_summaries),
        "category_performance": _build_category_performance(
            product_events=product_events,
            transactions=transactions,
        ),
        "note": (
            "Revenue, gross margin, and product rankings use filtered "
            "transactions. Current price, stockout rate, view activity, and "
            "elasticity use product-level data and are not region-specific."
        ),
    }


@app.get("/api/pricing-lab")
def get_pricing_lab(
    product_id: str | None = None,
    selected_price: float | None = None,
) -> dict[str, object]:
    """Return product pricing scenarios and optimizer markers."""

    data = load_pricing_data()
    product_ids = sorted(data.product_baselines.keys())
    if not product_ids:
        raise HTTPException(status_code=404, detail="No products available.")

    selected_product_id = product_id or product_ids[0]
    if selected_product_id not in data.product_baselines:
        raise HTTPException(status_code=404, detail=f"Unknown product: {selected_product_id}")

    baseline = data.product_baselines[selected_product_id]
    proposed_price = selected_price or baseline.current_average_selling_price
    proposed_price = round(proposed_price, 2)

    try:
        selected_scenario = evaluate_product_price_scenario(
            product_id=selected_product_id,
            proposed_price=proposed_price,
            data=data,
            scenario_label="Selected price",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    price_grid = build_product_price_grid(baseline)
    curve_scenarios = evaluate_product_price_scenarios(
        product_id=selected_product_id,
        proposed_prices=price_grid,
        data=data,
    )
    optima = find_product_optimal_prices(selected_product_id, data)
    recommendation = build_product_price_recommendation(selected_product_id, data)

    return {
        "products": _build_pricing_product_options(data),
        "product": asdict(baseline),
        "selected_scenario": asdict(selected_scenario),
        "scenarios": scenario_results_to_dicts(curve_scenarios),
        "recommendation": product_recommendation_to_dict(recommendation),
        "markers": {
            "current_price": baseline.current_average_selling_price,
            "selected_price": selected_scenario.proposed_price,
            "revenue_optimal_price": optima.revenue_optimal_scenario.proposed_price,
            "recommended_price": recommendation.recommended_price,
            "revenue_optimum_boundary_limited": (
                optima.revenue_optimum_boundary_limited
            ),
            "recommended_boundary_limited": recommendation.margin_optimum_boundary_limited,
            "revenue_boundary_hit": _boundary_hit_label(
                optima.revenue_optimal_scenario.proposed_price,
                baseline.min_tested_price,
                baseline.max_tested_price,
            ),
            "recommended_boundary_hit": recommendation.margin_optimum_boundary_hit,
        },
        "supported_range": {
            "min": baseline.min_tested_price,
            "max": baseline.max_tested_price,
            "current": baseline.current_average_selling_price,
            "step": 0.01,
        },
        "note": (
            "Pricing Lab scenarios use the full product history and current "
            "optimizer output; global date, region, and category filters are "
            "not applied to these model-based simulations."
        ),
    }


@app.get("/api/customer-analytics")
def get_customer_analytics(
    start_date: str | None = None,
    end_date: str | None = None,
    region: str = "all",
    category: str = "all",
) -> dict[str, object]:
    """Return customer churn and repeat-purchase summaries supported by current data."""

    data = load_csv_data()
    date_range = _available_date_bounds(data)
    start_date, end_date = _clamp_date_range(start_date, end_date, date_range)
    transactions = _filter_transactions(
        data["transactions"],
        start_date=start_date,
        end_date=end_date,
        region=region,
        category=category,
    )
    customers = _customers_for_filtered_context(
        customers=data["customers"],
        transactions=transactions,
        region=region,
    )

    return {
        "customer_count": len(customers),
        "churn_rate": _churn_rate(customers),
        "segments": {
            "region": _group_customer_churn(customers, "customer_region"),
            "acquisition_channel": _group_customer_churn(customers, "acquisition_channel"),
            "tenure": _group_by_customer_bucket(
                customers,
                field="customer_tenure_days",
                buckets=[
                    ("0-90 days", 0, 90),
                    ("91-180 days", 91, 180),
                    ("181-365 days", 181, 365),
                    ("366+ days", 366, math.inf),
                ],
            ),
            "purchase_frequency": _group_by_customer_bucket(
                customers,
                field="purchase_frequency",
                buckets=[
                    ("0-2 purchases/year", 0, 2),
                    ("2-4 purchases/year", 2, 4),
                    ("4-6 purchases/year", 4, 6),
                    ("6+ purchases/year", 6, math.inf),
                ],
            ),
        },
        "cohorts": _build_repeat_purchase_cohorts(
            customers=customers,
            transactions=transactions,
            all_transactions=data["transactions"],
            observation_start=start_date,
            observation_end=end_date,
            data_end=date_range["max"],
        ),
        "note": (
            "Customer filters define the buyer scope through matching "
            "transactions. Churn is a customer-level outcome; cohort cells show "
            "repeat-purchase activity after the acquisition transaction, not "
            "churn-derived retention."
        ),
    }


@app.get("/api/experimentation")
def get_experimentation() -> dict[str, object]:
    """Return A/B-style and causal pricing analysis summaries."""

    customers = load_csv_data()["customers"]
    ab_result = _build_ab_pricing_summary(customers)

    return {
        "ab_test": ab_result,
        "causal": {
            "treatment": "Price increase exposure",
            "outcome": "Customer churn",
            "adjusted_ipw_effect": 0.0245,
            "bootstrap_ci": [0.0131, 0.0369],
            "naive_difference": 0.0216,
            "discount_adjusted_sensitivity": 0.0255,
            "random_common_cause_change": -0.0001,
            "interpretation": (
                "The adjusted IPW estimate is about 0.29 percentage points "
                "larger than the current naive churn difference."
            ),
            "assumption_note": (
                "Causal estimates rely on observed synthetic confounders: "
                "purchase frequency, prior spending, tenure, region, channel, "
                "and discount usage."
            ),
        },
        "result_types": [
            {
                "label": "Descriptive",
                "description": "Observed summaries from the synthetic datasets.",
            },
            {
                "label": "Experimental-style",
                "description": "Treatment/control comparison by price exposure.",
            },
            {
                "label": "Causal",
                "description": "IPW adjustment using observed confounders.",
            },
            {
                "label": "Predictive",
                "description": "Churn model performance only, not a pricing engine.",
            },
        ],
    }


@app.get("/api/model-performance")
def get_model_performance() -> dict[str, object]:
    """Return current churn model metrics from the regenerated dataset."""

    return {
        "models": [
            {
                "model": "Logistic Regression",
                "role": "Baseline",
                "roc_auc": 0.533,
                "precision": 0.080,
                "recall": 0.502,
                "f1": 0.137,
                "confusion_matrix": [[1580, 1318], [113, 114]],
            },
            {
                "model": "Random Forest",
                "role": "Nonlinear comparison",
                "roc_auc": 0.524,
                "precision": 0.084,
                "recall": 0.515,
                "f1": 0.144,
                "confusion_matrix": [[1617, 1281], [110, 117]],
            },
        ],
        "finding": (
            "Prediction performance is modest. These models help explain churn "
            "risk patterns but are not the main pricing decision engine."
        ),
    }


@app.get("/")
def get_index() -> FileResponse:
    """Serve the dashboard entry point."""

    return FileResponse(FRONTEND_DIR / "index.html")


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _available_date_bounds(data: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    """Return the available synthetic data window across date-bearing files."""

    dates = []
    dates.extend(row["first_purchase_date"] for row in data["customers"])
    dates.extend(row["last_transaction_date"] for row in data["customers"])
    dates.extend(row["transaction_date"] for row in data["transactions"])
    dates.extend(row["event_date"] for row in data["product_events"])

    return {
        "min": min(dates),
        "max": max(dates),
    }


def _clamp_date_range(
    start_date: str | None,
    end_date: str | None,
    date_range: dict[str, str],
) -> tuple[str, str]:
    """Keep requested dashboard dates inside the available synthetic data window."""

    min_date = date_range["min"]
    max_date = date_range["max"]
    bounded_start = _clamp_iso_date(start_date or min_date, min_date, max_date)
    bounded_end = _clamp_iso_date(end_date or max_date, min_date, max_date)

    if bounded_start > bounded_end:
        bounded_end = bounded_start

    return bounded_start, bounded_end


def _clamp_iso_date(value: str, min_date: str, max_date: str) -> str:
    """Clamp an ISO date string without changing valid in-range dates."""

    if value < min_date:
        return min_date
    if value > max_date:
        return max_date
    return value


def _filter_transactions(
    transactions: Iterable[dict[str, str]],
    start_date: str | None,
    end_date: str | None,
    region: str,
    category: str,
) -> list[dict[str, str]]:
    """Filter transaction rows by supported dashboard controls."""

    return [
        row
        for row in transactions
        if _date_in_range(row["transaction_date"], start_date, end_date)
        and (region == "all" or row["customer_region"] == region)
        and (category == "all" or row["product_category"] == category)
    ]


def _filter_product_events(
    product_events: Iterable[dict[str, str]],
    start_date: str | None,
    end_date: str | None,
    category: str,
) -> list[dict[str, str]]:
    """Filter product-day event rows by supported product controls."""

    return [
        row
        for row in product_events
        if _date_in_range(row["event_date"], start_date, end_date)
        and (category == "all" or row["product_category"] == category)
    ]


def _date_in_range(
    value: str,
    start_date: str | None,
    end_date: str | None,
) -> bool:
    """Return whether an ISO date string is within optional bounds."""

    if start_date and value < start_date:
        return False
    if end_date and value > end_date:
        return False
    return True


def _month_number(month_text: str) -> int:
    """Convert YYYY-MM text into a sortable month index."""

    month_start = date.fromisoformat(f"{month_text}-01")
    return month_start.year * 12 + month_start.month


def _add_months(month_start: date, offset: int) -> date:
    """Return the first day of the month offset from another month start."""

    month_index = month_start.year * 12 + month_start.month - 1 + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def _month_end(month_start: date) -> date:
    """Return the final day of a calendar month."""

    return _add_months(month_start, 1) - timedelta(days=1)


def _customers_for_filtered_context(
    customers: Iterable[dict[str, str]],
    transactions: Iterable[dict[str, str]],
    region: str,
) -> list[dict[str, str]]:
    """Select customers represented by the current transaction filter."""

    customer_lookup = {row["customer_id"]: row for row in customers}
    filtered_customer_ids = {row["customer_id"] for row in transactions}
    selected = [
        customer_lookup[customer_id]
        for customer_id in filtered_customer_ids
        if customer_id in customer_lookup
    ]

    if region != "all":
        selected = [row for row in selected if row["customer_region"] == region]

    return selected


def _build_executive_kpis(
    transactions: list[dict[str, str]],
    customers: list[dict[str, str]],
) -> dict[str, float | int]:
    """Calculate executive KPI cards."""

    gmv = sum(_float(row["list_price"]) * _float(row["quantity"]) for row in transactions)
    net_revenue = sum(_float(row["order_value"]) for row in transactions)
    gross_margin = sum(_float(row["gross_margin"]) for row in transactions)
    total_orders = len(transactions)

    return {
        "gmv": gmv,
        "net_revenue": net_revenue,
        "gross_margin": gross_margin,
        "gross_margin_rate": _safe_divide(gross_margin, net_revenue),
        "aov": _safe_divide(net_revenue, total_orders),
        "churn_rate": _churn_rate(customers),
        "total_orders": total_orders,
        "customer_count": len(customers),
    }


def _build_funnel_summary(
    product_events: list[dict[str, str]],
    region: str,
) -> dict[str, object]:
    """Calculate conversion funnel metrics using product-day events."""

    views = sum(_int(row["product_views"]) for row in product_events)
    add_to_cart = sum(_int(row["add_to_cart_events"]) for row in product_events)
    checkout_started = sum(_int(row["checkout_started_events"]) for row in product_events)
    purchases = sum(_int(row["purchases"]) for row in product_events)

    note = "Funnel data is product-day based."
    if region != "all":
        note += " Region filters apply to transactions/customers, not product events."

    return {
        "product_views": views,
        "add_to_cart_events": add_to_cart,
        "checkout_started_events": checkout_started,
        "purchases": purchases,
        "add_to_cart_rate": _safe_divide(add_to_cart, views),
        "checkout_start_rate": _safe_divide(checkout_started, add_to_cart),
        "purchase_after_checkout_rate": _safe_divide(purchases, checkout_started),
        "view_to_purchase_rate": _safe_divide(purchases, views),
        "note": note,
    }


def _build_monthly_trends(
    transactions: list[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate monthly GMV, revenue, margin, order count, and AOV."""

    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {"gmv": 0.0, "net_revenue": 0.0, "gross_margin": 0.0, "orders": 0.0}
    )
    for row in transactions:
        month = row["transaction_date"][:7]
        groups[month]["gmv"] += _float(row["list_price"]) * _float(row["quantity"])
        groups[month]["net_revenue"] += _float(row["order_value"])
        groups[month]["gross_margin"] += _float(row["gross_margin"])
        groups[month]["orders"] += 1

    return [
        {
            "month": month,
            "gmv": values["gmv"],
            "net_revenue": values["net_revenue"],
            "gross_margin": values["gross_margin"],
            "orders": int(values["orders"]),
            "aov": _safe_divide(values["net_revenue"], values["orders"]),
        }
        for month, values in sorted(groups.items())
    ]


def _build_product_summaries(
    product_events: list[dict[str, str]],
    transactions: list[dict[str, str]],
    optimizer_data,
) -> list[dict[str, object]]:
    """Build product analytics rows from transactions, events, and optimizer estimates."""

    event_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "product_name": "",
            "product_category": "",
            "event_units_sold": 0.0,
            "views": 0,
            "purchases": 0,
            "stockout_days": 0,
            "days": 0,
        }
    )

    for row in product_events:
        product_id = row["product_id"]
        group = event_groups[product_id]
        group["product_name"] = row["product_name"]
        group["product_category"] = row["product_category"]
        group["event_units_sold"] += _float(row["units_sold"])
        group["views"] += _int(row["product_views"])
        group["purchases"] += _int(row["purchases"])
        group["stockout_days"] += 1 if _bool(row["stockout_flag"]) else 0
        group["days"] += 1

    transaction_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "product_name": "",
            "product_category": "",
            "revenue": 0.0,
            "gross_margin": 0.0,
            "units_sold": 0.0,
            "orders": 0,
        }
    )
    for row in transactions:
        product_id = row["product_id"]
        group = transaction_groups[product_id]
        group["product_name"] = row["product_name"]
        group["product_category"] = row["product_category"]
        group["revenue"] += _float(row["order_value"])
        group["gross_margin"] += _float(row["gross_margin"])
        group["units_sold"] += _float(row["quantity"])
        group["orders"] += 1

    summaries = []
    product_ids = set(event_groups) | set(transaction_groups)
    for product_id in product_ids:
        event_group = event_groups.get(product_id, {})
        transaction_group = transaction_groups.get(product_id, {})
        baseline = optimizer_data.product_baselines.get(product_id)
        revenue = transaction_group.get("revenue", 0.0)
        gross_margin = transaction_group.get("gross_margin", 0.0)
        units = transaction_group.get("units_sold", 0.0)
        event_units = event_group.get("event_units_sold", 0.0)
        event_days = event_group.get("days", 0)
        product_name = (
            transaction_group.get("product_name")
            or event_group.get("product_name")
            or (baseline.product_name if baseline else "")
        )
        product_category = (
            transaction_group.get("product_category")
            or event_group.get("product_category")
            or (baseline.product_category if baseline else "")
        )
        summaries.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "product_category": product_category,
                "current_price": (
                    baseline.current_average_selling_price
                    if baseline
                    else _safe_divide(revenue, units)
                ),
                "unit_velocity": _safe_divide(event_units, event_days),
                "revenue": revenue,
                "gross_margin": gross_margin,
                "gross_margin_rate": _safe_divide(gross_margin, revenue),
                "stockout_rate": _safe_divide(
                    event_group.get("stockout_days", 0),
                    event_days,
                ),
                "estimated_elasticity": (
                    baseline.estimated_price_elasticity if baseline else 0.0
                ),
                "units_sold": units,
                "event_units_sold": event_units,
                "views": event_group.get("views", 0),
                "purchases": event_group.get("purchases", 0),
                "orders": transaction_group.get("orders", 0),
            }
        )

    return sorted(summaries, key=lambda row: row["revenue"], reverse=True)


def _build_top_bottom_products(
    products: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Return top and bottom product lists by revenue and gross margin."""

    by_revenue = sorted(products, key=lambda row: row["revenue"], reverse=True)
    by_margin = sorted(products, key=lambda row: row["gross_margin"], reverse=True)

    return {
        "top_revenue": by_revenue[:5],
        "bottom_revenue": list(reversed(by_revenue[-5:])),
        "top_margin": by_margin[:5],
        "bottom_margin": list(reversed(by_margin[-5:])),
    }


def _build_category_performance(
    product_events: list[dict[str, str]],
    transactions: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Aggregate category transaction economics and product-day stockouts."""

    event_groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "stockout_days": 0.0,
            "days": 0.0,
        }
    )
    for row in product_events:
        category = row["product_category"]
        event_groups[category]["stockout_days"] += 1 if _bool(row["stockout_flag"]) else 0
        event_groups[category]["days"] += 1

    transaction_groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {"revenue": 0.0, "gross_margin": 0.0, "units_sold": 0.0}
    )
    for row in transactions:
        category = row["product_category"]
        transaction_groups[category]["revenue"] += _float(row["order_value"])
        transaction_groups[category]["gross_margin"] += _float(row["gross_margin"])
        transaction_groups[category]["units_sold"] += _float(row["quantity"])

    return [
        {
            "category": category,
            "revenue": transaction_groups[category]["revenue"],
            "gross_margin": transaction_groups[category]["gross_margin"],
            "gross_margin_rate": _safe_divide(
                transaction_groups[category]["gross_margin"],
                transaction_groups[category]["revenue"],
            ),
            "units_sold": transaction_groups[category]["units_sold"],
            "stockout_rate": _safe_divide(
                event_groups[category]["stockout_days"],
                event_groups[category]["days"],
            ),
        }
        for category, values in sorted(
            transaction_groups.items(),
            key=lambda item: item[1]["revenue"],
            reverse=True,
        )
    ]


def _build_pricing_product_options(optimizer_data) -> list[dict[str, object]]:
    """Return product options for the Pricing Lab selector."""

    return [
        {
            "product_id": baseline.product_id,
            "product_name": baseline.product_name,
            "product_category": baseline.product_category,
            "current_price": baseline.current_average_selling_price,
        }
        for baseline in sorted(
            optimizer_data.product_baselines.values(),
            key=lambda item: item.product_name,
        )
    ]


def _group_customer_churn(
    customers: Iterable[dict[str, str]],
    field: str,
) -> list[dict[str, object]]:
    """Group churn rate by a customer field."""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in customers:
        groups[row[field]].append(row)

    return [
        {
            "segment": segment,
            "customers": len(rows),
            "churn_rate": _churn_rate(rows),
            "retention_rate": 1 - _churn_rate(rows),
        }
        for segment, rows in sorted(groups.items())
    ]


def _group_by_customer_bucket(
    customers: Iterable[dict[str, str]],
    field: str,
    buckets: list[tuple[str, float, float]],
) -> list[dict[str, object]]:
    """Group customers into numeric buckets and calculate churn."""

    groups = {label: [] for label, _, _ in buckets}
    for row in customers:
        value = _float(row[field])
        for label, lower, upper in buckets:
            if lower <= value < upper or (upper == math.inf and value >= lower):
                groups[label].append(row)
                break

    return [
        {
            "segment": label,
            "customers": len(rows),
            "churn_rate": _churn_rate(rows),
            "retention_rate": 1 - _churn_rate(rows),
        }
        for label, rows in groups.items()
    ]


def _build_repeat_purchase_cohorts(
    customers: list[dict[str, str]],
    transactions: list[dict[str, str]],
    all_transactions: list[dict[str, str]],
    observation_start: str,
    observation_end: str,
    data_end: str,
) -> dict[str, object]:
    """Build mature repeat-purchase activity by cohort and month since acquisition."""

    if not customers:
        return {
            "months": [],
            "rows": [],
            "metric_label": "Repeat-purchase rate",
            "definition": (
                "M0 is the acquisition calendar month after excluding each "
                "customer's first transaction; M1 is the next calendar month, "
                "and so on."
            ),
            "footnote": "No customers matched the current filter scope.",
            "observation_window": {
                "start": observation_start,
                "end": observation_end,
            },
        }

    customer_lookup = {row["customer_id"]: row for row in customers}
    cohort_customers: dict[str, set[str]] = defaultdict(set)
    for customer_id, customer in customer_lookup.items():
        cohort_customers[customer["first_purchase_date"][:7]].add(customer_id)

    first_transaction_by_customer = _first_transaction_ids(all_transactions)
    observation_start_date = date.fromisoformat(observation_start)
    observation_end_date = date.fromisoformat(observation_end)
    data_end_date = date.fromisoformat(data_end)

    active_by_cohort_month: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in transactions:
        customer_id = row["customer_id"]
        customer = customer_lookup.get(customer_id)
        if not customer:
            continue
        if row["transaction_id"] == first_transaction_by_customer.get(customer_id):
            continue

        cohort = customer["first_purchase_date"][:7]
        month_offset = (
            _month_number(row["transaction_date"][:7]) - _month_number(cohort)
        )
        if 0 <= month_offset <= 12:
            active_by_cohort_month[(cohort, month_offset)].add(customer_id)

    month_offsets = list(range(13))
    rows = []
    for cohort, customer_ids in sorted(cohort_customers.items()):
        cohort_month_start = date.fromisoformat(f"{cohort}-01")
        cohort_size = len(customer_ids)
        cells = []
        for month_offset in month_offsets:
            period_start = _add_months(cohort_month_start, month_offset)
            period_end = _month_end(period_start)
            eligible = (
                period_start >= observation_start_date
                and period_end <= observation_end_date
                and period_end <= data_end_date
            )
            active_customers = len(active_by_cohort_month[(cohort, month_offset)])
            cells.append(
                {
                    "month": month_offset,
                    "eligible": eligible,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "active_customers": active_customers if eligible else None,
                    "repeat_purchase_rate": (
                        _safe_divide(active_customers, cohort_size)
                        if eligible
                        else None
                    ),
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "customers": cohort_size,
                "cells": cells,
            }
        )

    display_rows = [row for row in rows if any(cell["eligible"] for cell in row["cells"])]
    display_rows = display_rows[-12:]
    mature_months = [
        month_offset
        for month_offset in month_offsets
        if any(
            cell["month"] == month_offset and cell["eligible"]
            for row in display_rows
            for cell in row["cells"]
        )
    ]
    for row in display_rows:
        row["cells"] = [
            cell for cell in row["cells"] if cell["month"] in mature_months
        ]

    return {
        "months": mature_months,
        "rows": display_rows,
        "metric_label": "Repeat-purchase rate",
        "definition": (
            "M0 is the acquisition calendar month after excluding each "
            "customer's first transaction; M1 is the next calendar month, "
            "and so on."
        ),
        "footnote": (
            "Denominators are customers in the selected customer scope for "
            "each acquisition month. Cells appear only when the full calendar "
            f"month is inside the selected date window ({observation_start} "
            f"to {observation_end}) and completed by the dataset end date "
            f"({data_end}); -- marks immature or partial periods."
        ),
        "observation_window": {
            "start": observation_start,
            "end": observation_end,
        },
    }


def _first_transaction_ids(
    transactions: list[dict[str, str]],
) -> dict[str, str]:
    """Return each customer's earliest transaction ID."""

    first_transactions = {}
    for row in sorted(
        transactions,
        key=lambda item: (
            item["customer_id"],
            item["transaction_date"],
            item["transaction_id"],
        ),
    ):
        first_transactions.setdefault(row["customer_id"], row["transaction_id"])

    return first_transactions


def _build_ab_pricing_summary(
    customers: list[dict[str, str]],
) -> dict[str, object]:
    """Calculate two-proportion A/B-style churn comparison."""

    treatment = [row for row in customers if _bool(row["price_increase_occurred"])]
    control = [row for row in customers if not _bool(row["price_increase_occurred"])]
    treatment_churned = sum(1 for row in treatment if _bool(row["churned"]))
    control_churned = sum(1 for row in control if _bool(row["churned"]))
    treatment_rate = _safe_divide(treatment_churned, len(treatment))
    control_rate = _safe_divide(control_churned, len(control))
    difference = treatment_rate - control_rate
    relative_difference = _safe_divide(difference, control_rate)

    standard_error = math.sqrt(
        _safe_divide(treatment_rate * (1 - treatment_rate), len(treatment))
        + _safe_divide(control_rate * (1 - control_rate), len(control))
    )
    pooled_rate = _safe_divide(treatment_churned + control_churned, len(treatment) + len(control))
    pooled_error = math.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (_safe_divide(1, len(treatment)) + _safe_divide(1, len(control)))
    )
    z_score = _safe_divide(difference, pooled_error)
    p_value = math.erfc(abs(z_score) / math.sqrt(2))

    return {
        "control": {
            "label": "Control: no price increase exposure",
            "customers": len(control),
            "churn_rate": control_rate,
        },
        "treatment": {
            "label": "Treatment: price increase exposure",
            "customers": len(treatment),
            "churn_rate": treatment_rate,
        },
        "absolute_difference": difference,
        "relative_difference": relative_difference,
        "confidence_interval": [
            difference - 1.96 * standard_error,
            difference + 1.96 * standard_error,
        ],
        "p_value": p_value,
        "note": (
            "This is an experimental-style comparison using synthetic exposure "
            "groups; it is not by itself a causal claim."
        ),
    }


def _boundary_hit_label(price: float, lower: float, upper: float) -> str:
    """Label boundary hits for frontend badges."""

    tolerance = 0.005
    if abs(price - lower) <= tolerance:
        return "lower"
    if abs(price - upper) <= tolerance:
        return "upper"
    return "interior"


def _churn_rate(customers: Iterable[dict[str, str]]) -> float:
    """Calculate churn rate for a customer collection."""

    rows = list(customers)
    return _safe_divide(sum(1 for row in rows if _bool(row["churned"])), len(rows))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely when a denominator may be zero."""

    return numerator / denominator if denominator else 0.0


def _float(value: Any) -> float:
    """Parse CSV numeric values as floats."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    """Parse CSV numeric values as ints."""

    return int(_float(value))


def _bool(value: Any) -> bool:
    """Parse CSV boolean values."""

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIR),
    name="assets",
)
