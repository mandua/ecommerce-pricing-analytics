import {
    cssTone,
    formatBoundaryHit,
    formatCategory,
    formatCompactNumber,
    formatCurrency,
    formatDateLabel,
    formatMetric,
    formatNumber,
    formatPercent,
    formatPValue,
    formatPrice,
    formatSignedCurrency,
    formatSignedPercent,
} from "./formatters.js";


export function setStatus(element, type, message) {
    element.className = `status-message ${type || ""}`.trim();
    element.textContent = message;
}


export function populateSelect(element, rows, selectedValue, options = {}) {
    const {
        valueKey = "value",
        labelKey = "label",
        includeAll = false,
        allLabel = "All",
    } = options;

    const allOption = includeAll ? `<option value="all">${allLabel}</option>` : "";
    element.innerHTML =
        allOption +
        rows
            .map((row) => {
                const value = row[valueKey] ?? row;
                const label = row[labelKey] ?? row;
                return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
            })
            .join("");
    element.value = selectedValue || (includeAll ? "all" : element.options[0]?.value);
}


export function renderExecutiveDashboard(container, executive) {
    const { kpis, funnel, monthly_trends: monthlyTrends } = executive;
    container.innerHTML = `
        <div class="kpi-grid executive-grid">
            ${metricCard("GMV", formatCurrency(kpis.gmv), "List price x units")}
            ${metricCard("Net Revenue", formatCurrency(kpis.net_revenue), "After synthetic discounts")}
            ${metricCard("Gross Margin", formatCurrency(kpis.gross_margin), `${formatPercent(kpis.gross_margin_rate)} margin rate`)}
            ${metricCard("AOV", formatCurrency(kpis.aov), "Net revenue per order")}
            ${metricCard("Churn", formatPercent(kpis.churn_rate), `${formatNumber(kpis.customer_count)} customers in scope`)}
            ${metricCard("Total Orders", formatNumber(kpis.total_orders), "Transaction rows")}
        </div>

        <div class="dashboard-grid two-column">
            <section class="panel">
                <div class="section-heading compact">
                    <p class="eyebrow">Conversion Funnel</p>
                    <h2>Product-Day Funnel Summary</h2>
                </div>
                ${renderFunnel(funnel)}
                <p class="supporting-copy">${escapeHtml(funnel.note)}</p>
            </section>

            <section class="panel">
                <div class="section-heading compact">
                    <p class="eyebrow">Trend</p>
                    <h2>Revenue and Margin by Month</h2>
                </div>
                ${lineChart(monthlyTrends, [
                    { key: "net_revenue", label: "Net revenue", color: "cyan" },
                    { key: "gross_margin", label: "Gross margin", color: "green" },
                ], "month", "Monthly ecommerce economics")}
            </section>
        </div>
    `;
}


export function renderProductAnalytics(container, payload) {
    const selected = payload.selected_product;
    if (!selected) {
        container.innerHTML = emptyState("No product activity for the selected filters.");
        return;
    }

    container.innerHTML = `
        <div class="kpi-grid product-grid">
            ${metricCard("Current Price", formatPrice(selected.current_price), selected.product_name)}
            ${metricCard("Unit Velocity", `${formatNumber(selected.unit_velocity, 2)}/day`, "Product-day demand")}
            ${metricCard("Revenue", formatCurrency(selected.revenue), `${formatNumber(selected.orders)} filtered orders`)}
            ${metricCard("Gross Margin", formatCurrency(selected.gross_margin), `${formatPercent(selected.gross_margin_rate)} transaction margin rate`)}
            ${metricCard("Stockout Rate", formatPercent(selected.stockout_rate), "Product-day stockouts, not regional")}
            ${metricCard("Elasticity", formatMetric(selected.estimated_elasticity, 2), "Full-history optimizer estimate")}
        </div>

        <div class="dashboard-grid three-column">
            ${productRankPanel("Top Revenue Products", payload.top_bottom.top_revenue, "revenue")}
            ${productRankPanel("Bottom Revenue Products", payload.top_bottom.bottom_revenue, "revenue")}
            ${productRankPanel("Top Margin Products", payload.top_bottom.top_margin, "gross_margin")}
        </div>

        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Category Performance</p>
                <h2>Revenue, Margin, Velocity, and Stockouts</h2>
            </div>
            ${categoryTable(payload.category_performance)}
            <p class="supporting-copy">${escapeHtml(payload.note)}</p>
        </section>
    `;
}


export function renderPricingLab(container, payload) {
    const { product, recommendation, selected_scenario: selected, scenarios, markers } =
        payload;
    const revenueBoundaryNote = markers.revenue_optimum_boundary_limited
        ? `Revenue optimum hit the ${formatBoundaryHit(markers.revenue_boundary_hit).toLowerCase()}.`
        : "Revenue optimum is inside the tested range.";

    container.innerHTML = `
        <div class="pricing-summary">
            <article class="recommendation-card">
                <span class="eyebrow">Recommended Price</span>
                <strong>${formatPrice(recommendation.recommended_price)}</strong>
                <small>${formatSignedPercent(recommendation.recommended_percent_change)} vs current, optimized for gross margin</small>
                <p>${escapeHtml(recommendation.recommendation_basis)}</p>
                ${
                    recommendation.margin_optimum_boundary_limited
                        ? `<p class="warning-text">Recommended price is boundary-limited at ${formatBoundaryHit(recommendation.margin_optimum_boundary_hit)}.</p>`
                        : ""
                }
            </article>
            <article class="comparison-card">
                <span class="eyebrow">Revenue-Only Comparison</span>
                <strong>${formatPrice(recommendation.revenue_optimum.price)}</strong>
                <small>${formatSignedPercent(recommendation.revenue_optimum.percent_change)} vs current</small>
                <p>${revenueBoundaryNote}</p>
            </article>
        </div>

        <div class="kpi-grid pricing-grid">
            ${metricCard("Expected Units", formatNumber(selected.expected_units_sold), `${formatNumber(selected.expected_unit_velocity, 2)}/day`)}
            ${metricCard("Expected Revenue", formatCurrency(selected.expected_revenue), `${formatSignedCurrency(selected.revenue_change_vs_baseline)} vs baseline`, cssTone(selected.revenue_change_vs_baseline))}
            ${metricCard("Expected Gross Margin", formatCurrency(selected.expected_gross_margin), `${formatSignedCurrency(selected.margin_change_vs_baseline)} vs baseline`, cssTone(selected.margin_change_vs_baseline))}
            ${metricCard("Estimated Churn", formatPercent(selected.churn_impact.estimated_churn_rate), `${formatSignedPercent(selected.churn_impact.churn_rate_change)} vs product baseline`, cssTone(-selected.churn_impact.churn_rate_change))}
        </div>

        <div class="dashboard-grid three-column curve-grid">
            ${curvePanel("Demand Curve", "Expected units sold", scenarios, selected, markers, "expected_units_sold")}
            ${curvePanel("Revenue Curve", "Expected net revenue", scenarios, selected, markers, "expected_revenue")}
            ${curvePanel("Gross-Margin Curve", "Expected gross margin", scenarios, selected, markers, "expected_gross_margin")}
        </div>

        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Product Context</p>
                <h2>${escapeHtml(product.product_name)}</h2>
            </div>
            <div class="detail-grid">
                ${detailItem("Category", formatCategory(product.product_category))}
                ${detailItem("Current price", formatPrice(product.current_average_selling_price))}
                ${detailItem("Unit velocity", `${formatNumber(product.baseline_unit_velocity, 2)}/day`)}
                ${detailItem("Elasticity", formatMetric(product.estimated_price_elasticity, 2))}
                ${detailItem("Stockout rate", formatPercent(product.stockout_day_rate))}
                ${detailItem("Buyer churn", formatPercent(product.baseline_product_churn_rate))}
            </div>
            <p class="supporting-copy">${escapeHtml(payload.note)}</p>
        </section>
    `;
}


export function renderCustomerAnalytics(container, payload) {
    container.innerHTML = `
        <div class="kpi-grid compact-grid">
            ${metricCard("Customers", formatNumber(payload.customer_count), "Filtered customer scope")}
            ${metricCard("Churn Rate", formatPercent(payload.churn_rate), "Observed synthetic outcome")}
        </div>

        <div class="dashboard-grid two-column">
            ${segmentPanel("Churn by Region", payload.segments.region)}
            ${segmentPanel("Churn by Acquisition Channel", payload.segments.acquisition_channel)}
            ${segmentPanel("Churn by Tenure", payload.segments.tenure)}
            ${segmentPanel("Churn by Purchase Frequency", payload.segments.purchase_frequency)}
        </div>

        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Cohort Activity</p>
                <h2>Repeat Purchase by Acquisition Month</h2>
                <span>${escapeHtml(payload.cohorts.definition)}</span>
            </div>
            ${cohortHeatmap(payload.cohorts)}
            <p class="supporting-copy">${escapeHtml(payload.cohorts.footnote)}</p>
            <p class="supporting-copy">${escapeHtml(payload.note)}</p>
        </section>
    `;
}


export function renderExperimentation(container, payload) {
    const { ab_test: abTest, causal, result_types: resultTypes } = payload;
    container.innerHTML = `
        <div class="dashboard-grid two-column">
            <section class="panel">
                <div class="section-heading compact">
                    <p class="eyebrow">Experimental-Style</p>
                    <h2>Price Exposure and Churn</h2>
                </div>
                <div class="experiment-grid">
                    ${metricCard("Control Churn", formatPercent(abTest.control.churn_rate), `${formatNumber(abTest.control.customers)} customers`)}
                    ${metricCard("Treatment Churn", formatPercent(abTest.treatment.churn_rate), `${formatNumber(abTest.treatment.customers)} customers`)}
                    ${metricCard("Difference", formatSignedPercent(abTest.absolute_difference), `${formatSignedPercent(abTest.relative_difference)} relative`)}
                    ${metricCard("P-Value", formatPValue(abTest.p_value), "Two-proportion test")}
                </div>
                <p class="supporting-copy">
                    95% CI: <strong>${formatSignedPercent(abTest.confidence_interval[0])}</strong>
                    to <strong>${formatSignedPercent(abTest.confidence_interval[1])}</strong>.
                    ${escapeHtml(abTest.note)}
                </p>
            </section>

            <section class="panel">
                <div class="section-heading compact">
                    <p class="eyebrow">Causal</p>
                    <h2>Adjusted Price Effect</h2>
                </div>
                <div class="detail-grid">
                    ${detailItem("Treatment", causal.treatment)}
                    ${detailItem("Outcome", causal.outcome)}
                    ${detailItem("IPW estimate", formatSignedPercent(causal.adjusted_ipw_effect))}
                    ${detailItem("Bootstrap CI", `${formatSignedPercent(causal.bootstrap_ci[0])} to ${formatSignedPercent(causal.bootstrap_ci[1])}`)}
                    ${detailItem("Naive difference", formatSignedPercent(causal.naive_difference))}
                    ${detailItem("Robustness", `${formatSignedPercent(causal.random_common_cause_change, 2)} random common cause change`)}
                </div>
                <p class="supporting-copy">${escapeHtml(causal.interpretation)}</p>
                <p class="supporting-copy">${escapeHtml(causal.assumption_note)}</p>
            </section>
        </div>

        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Result Types</p>
                <h2>How to Read the Evidence</h2>
            </div>
            <div class="type-grid">
                ${resultTypes
                    .map(
                        (item) => `
                            <article class="type-card">
                                <strong>${escapeHtml(item.label)}</strong>
                                <p>${escapeHtml(item.description)}</p>
                            </article>
                        `
                    )
                    .join("")}
            </div>
        </section>
    `;
}


export function renderModelPerformance(container, payload) {
    container.innerHTML = `
        <section class="panel model-caveat">
            <span class="eyebrow">Exploratory Only</span>
            <p>ROC-AUC is about 0.53, so these churn models are useful context but are not the primary pricing decision engine.</p>
        </section>
        <div class="dashboard-grid two-column">
            ${payload.models.map((model) => modelCard(model)).join("")}
        </div>
        <section class="panel insight-panel">
            <span class="eyebrow">Model Readout</span>
            <p>${escapeHtml(payload.finding)}</p>
        </section>
    `;
}


function metricCard(label, value, detail, tone = "neutral") {
    return `
        <article class="metric-card ${tone}">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(detail)}</small>
        </article>
    `;
}


function detailItem(label, value) {
    return `
        <div class="detail-item">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `;
}


function renderFunnel(funnel) {
    const stages = [
        ["Product Views", funnel.product_views, 1],
        ["Add to Cart", funnel.add_to_cart_events, funnel.add_to_cart_rate],
        ["Checkout Started", funnel.checkout_started_events, funnel.checkout_start_rate],
        ["Purchases", funnel.purchases, funnel.purchase_after_checkout_rate],
    ];
    const maxValue = Math.max(...stages.map((stage) => stage[1]), 1);

    return `
        <div class="funnel-list">
            ${stages
                .map(([label, count, rate], index) => {
                    const width = Math.max((count / maxValue) * 100, 3);
                    const rateLabel = index === 0 ? "100.0%" : formatPercent(rate);
                    return `
                        <div class="funnel-row">
                            <div>
                                <strong>${escapeHtml(label)}</strong>
                                <span>${formatCompactNumber(count)} events</span>
                            </div>
                            <div class="bar-track">
                                <span style="width: ${width}%"></span>
                            </div>
                            <em>${rateLabel}</em>
                        </div>
                    `;
                })
                .join("")}
        </div>
    `;
}


function productRankPanel(title, rows, metric) {
    return `
        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Products</p>
                <h2>${escapeHtml(title)}</h2>
            </div>
            ${rankList(rows, metric)}
        </section>
    `;
}


function rankList(rows, metric) {
    const maxValue = Math.max(...rows.map((row) => row[metric]), 1);
    return `
        <div class="rank-list">
            ${rows
                .map(
                    (row) => `
                        <article class="rank-row">
                            <div>
                                <strong>${escapeHtml(row.product_name)}</strong>
                                <span>${formatCategory(row.product_category)}</span>
                            </div>
                            <div class="mini-bar">
                                <span style="width: ${Math.max((row[metric] / maxValue) * 100, 4)}%"></span>
                            </div>
                            <em>${formatCurrency(row[metric])}</em>
                        </article>
                    `
                )
                .join("")}
        </div>
    `;
}


function categoryTable(rows) {
    return `
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Revenue</th>
                        <th>Gross Margin</th>
                        <th>Units</th>
                        <th>Stockout</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows
                        .map(
                            (row) => `
                                <tr>
                                    <td>${formatCategory(row.category)}</td>
                                    <td>${formatCurrency(row.revenue)}</td>
                                    <td>${formatCurrency(row.gross_margin)} <span>${formatPercent(row.gross_margin_rate)}</span></td>
                                    <td>${formatNumber(row.units_sold)}</td>
                                    <td>${formatPercent(row.stockout_rate)}</td>
                                </tr>
                            `
                        )
                        .join("")}
                </tbody>
            </table>
        </div>
    `;
}


function segmentPanel(title, rows) {
    return `
        <section class="panel">
            <div class="section-heading compact">
                <p class="eyebrow">Customers</p>
                <h2>${escapeHtml(title)}</h2>
            </div>
            ${segmentBars(rows)}
        </section>
    `;
}


function segmentBars(rows) {
    const maxRate = Math.max(...rows.map((row) => row.churn_rate), 0.01);
    return `
        <div class="segment-list">
            ${rows
                .map(
                    (row) => `
                        <article class="segment-row">
                            <div>
                                <strong>${escapeHtml(formatCategory(row.segment))}</strong>
                                <span>${formatNumber(row.customers)} customers</span>
                            </div>
                            <div class="bar-track red">
                                <span style="width: ${Math.max((row.churn_rate / maxRate) * 100, 3)}%"></span>
                            </div>
                            <em>${formatPercent(row.churn_rate)}</em>
                        </article>
                    `
                )
                .join("")}
        </div>
    `;
}


function cohortHeatmap(cohorts) {
    const rows = cohorts.rows || [];
    const monthColumns = Array.from({ length: 13 }, (_, month) => month);
    if (!rows.length) {
        return emptyState("No repeat-purchase cohort data for the selected filters.");
    }

    return `
        <div class="heatmap-wrap">
            <div class="cohort-heatmap" role="table" aria-label="Repeat-purchase cohort heatmap">
                <span class="heatmap-header">Cohort</span>
                <span class="heatmap-header">Size</span>
                ${monthColumns
                    .map((month) => `<span class="heatmap-header">M${month}</span>`)
                    .join("")}
                ${rows
                    .map((row) => {
                        const cellsByMonth = new Map(
                            row.cells.map((cell) => [cell.month, cell])
                        );
                        const cells = monthColumns
                            .map((month) => {
                                const cell = cellsByMonth.get(month);
                                if (!cell || !cell.eligible) {
                                    return `<span class="heatmap-cell unavailable">--</span>`;
                                }
                                const rate = Number(cell.repeat_purchase_rate) || 0;
                                const heat = Math.min(rate / 0.2, 1).toFixed(3);
                                return `
                                    <span
                                        class="heatmap-cell"
                                        style="--heat: ${heat}"
                                        title="${formatPercent(rate)} repeat-purchase rate (${formatNumber(cell.active_customers)} of ${formatNumber(row.customers)})"
                                    >
                                        ${formatPercent(rate, 0)}
                                    </span>
                                `;
                            })
                            .join("");
                        return `
                            <span class="heatmap-row-label">${formatDateLabel(row.cohort)}</span>
                            <span class="heatmap-row-size">${formatNumber(row.customers)}</span>
                            ${cells}
                        `;
                    })
                    .join("")}
            </div>
            <div class="heatmap-legend" aria-hidden="true">
                <span>Lower</span>
                <i></i>
                <span>Higher repeat purchase</span>
            </div>
        </div>
    `;
}


function modelCard(model) {
    return `
        <section class="panel model-card">
            <div class="section-heading compact">
                <p class="eyebrow">${escapeHtml(model.role)}</p>
                <h2>${escapeHtml(model.model)}</h2>
            </div>
            <div class="model-metrics">
                ${detailItem("ROC-AUC", formatMetric(model.roc_auc, 3))}
                ${detailItem("Precision", formatMetric(model.precision, 3))}
                ${detailItem("Recall", formatMetric(model.recall, 3))}
                ${detailItem("F1", formatMetric(model.f1, 3))}
            </div>
            ${confusionMatrix(model.confusion_matrix)}
        </section>
    `;
}


function confusionMatrix(matrix) {
    return `
        <div class="confusion-grid" aria-label="Confusion matrix">
            <span>TN<br><strong>${formatNumber(matrix[0][0])}</strong></span>
            <span>FP<br><strong>${formatNumber(matrix[0][1])}</strong></span>
            <span>FN<br><strong>${formatNumber(matrix[1][0])}</strong></span>
            <span>TP<br><strong>${formatNumber(matrix[1][1])}</strong></span>
        </div>
    `;
}


function curvePanel(title, description, scenarios, selected, markers, metric) {
    return `
        <section class="panel curve-panel">
            <div class="section-heading compact">
                <p class="eyebrow">Pricing Lab</p>
                <h2>${escapeHtml(title)}</h2>
                <span>${escapeHtml(description)}</span>
            </div>
            ${lineChart(
                scenarios,
                [{ key: metric, label: title, color: metric === "expected_gross_margin" ? "green" : "cyan" }],
                "proposed_price",
                title,
                {
                    selectedPrice: selected.proposed_price,
                    currentPrice: markers.current_price,
                    revenueOptimalPrice: markers.revenue_optimal_price,
                    recommendedPrice: markers.recommended_price,
                }
            )}
        </section>
    `;
}


function lineChart(rows, series, xKey, label, markers = null) {
    if (!rows.length) {
        return emptyState("No chart data available.");
    }

    const width = 760;
    const height = 280;
    const padding = { top: 24, right: 26, bottom: 42, left: 68 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const xValues = rows.map((row, index) => (xKey === "month" ? index : row[xKey]));
    const yValues = series.flatMap((item) => rows.map((row) => row[item.key]));
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const yPad = Math.max((yMax - yMin) * 0.14, yMax * 0.04, 1);
    const yDomainMin = Math.max(0, yMin - yPad);
    const yDomainMax = yMax + yPad;

    const xScale = (value) =>
        padding.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotWidth;
    const yScale = (value) =>
        padding.top + ((yDomainMax - value) / Math.max(yDomainMax - yDomainMin, 1)) * plotHeight;

    const gridTicks = [0, 0.5, 1].map(
        (ratio) => yDomainMin + (yDomainMax - yDomainMin) * ratio
    );
    const xTicks = xKey === "month"
        ? rows.filter((_, index) => index % Math.ceil(rows.length / 6) === 0)
        : [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]];

    return `
        <svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)}">
            ${gridTicks
                .map((tick) => {
                    const y = yScale(tick);
                    return `
                        <line class="chart-grid" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
                        <text class="chart-tick" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${formatCompactNumber(tick)}</text>
                    `;
                })
                .join("")}
            <line class="chart-axis" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}"></line>
            <line class="chart-axis" x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}"></line>
            ${series
                .map((item) => {
                    const path = rows
                        .map((row, index) => {
                            const xValue = xKey === "month" ? index : row[xKey];
                            const command = index === 0 ? "M" : "L";
                            return `${command}${xScale(xValue).toFixed(2)},${yScale(row[item.key]).toFixed(2)}`;
                        })
                        .join(" ");
                    return `<path class="chart-line ${item.color}" d="${path}"></path>`;
                })
                .join("")}
            ${xTicks
                .map((rowOrValue) => {
                    const xValue = xKey === "month" ? rows.indexOf(rowOrValue) : rowOrValue[xKey];
                    const labelText = xKey === "month"
                        ? formatDateLabel(rowOrValue.month)
                        : formatPrice(rowOrValue[xKey]);
                    return `<text class="chart-tick" x="${xScale(xValue)}" y="${height - 14}" text-anchor="middle">${labelText}</text>`;
                })
                .join("")}
            ${markers ? markerLayer(rows, series[0].key, xScale, yScale, markers) : ""}
        </svg>
        ${legend(series, markers)}
    `;
}


function markerLayer(rows, yKey, xScale, yScale, markers) {
    const markerDefs = [
        ["current", markers.currentPrice, "Current"],
        ["selected", markers.selectedPrice, "Selected"],
        ["revenue", markers.revenueOptimalPrice, "Revenue opt"],
        ["recommended", markers.recommendedPrice, "Recommended"],
    ];

    return markerDefs
        .map(([className, price, label]) => {
            const closest = rows.reduce((best, row) =>
                Math.abs(row.proposed_price - price) < Math.abs(best.proposed_price - price)
                    ? row
                    : best
            );
            const x = xScale(closest.proposed_price);
            const y = yScale(closest[yKey]);
            return `
                <line class="marker-line ${className}" x1="${x}" y1="24" x2="${x}" y2="238"></line>
                <circle class="chart-marker ${className}" cx="${x}" cy="${y}" r="5"></circle>
                <text class="marker-label ${className}" x="${x + 6}" y="${Math.max(y - 8, 14)}">${label}</text>
            `;
        })
        .join("");
}


function legend(series, markers) {
    const seriesLegend = series
        .map((item) => `<span><i class="${item.color}"></i>${escapeHtml(item.label)}</span>`)
        .join("");
    const markerLegend = markers
        ? `
            <span><i class="selected"></i>Selected</span>
            <span><i class="recommended"></i>Recommended</span>
            <span><i class="revenue"></i>Revenue opt</span>
        `
        : "";
    return `<div class="chart-legend">${seriesLegend}${markerLegend}</div>`;
}


function emptyState(message) {
    return `<div class="empty-state">${escapeHtml(message)}</div>`;
}


function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
