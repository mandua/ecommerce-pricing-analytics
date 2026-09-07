export function formatCurrency(value, maximumFractionDigits = 0) {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits,
    }).format(value || 0);
}


export function formatSignedCurrency(value, maximumFractionDigits = 0) {
    const numericValue = Math.abs(value) < 0.005 ? 0 : value;
    if (numericValue === 0) {
        return formatCurrency(0, maximumFractionDigits);
    }
    const sign = numericValue > 0 ? "+" : "-";
    return `${sign}${formatCurrency(Math.abs(numericValue), maximumFractionDigits)}`;
}


export function formatPercent(value, maximumFractionDigits = 1) {
    return new Intl.NumberFormat("en-US", {
        style: "percent",
        minimumFractionDigits: maximumFractionDigits,
        maximumFractionDigits,
    }).format(value || 0);
}


export function formatSignedPercent(value, maximumFractionDigits = 1) {
    const numericValue = Math.abs(value) < 0.0005 ? 0 : value;
    if (numericValue === 0) {
        return formatPercent(0, maximumFractionDigits);
    }
    const sign = numericValue > 0 ? "+" : "-";
    return `${sign}${formatPercent(Math.abs(numericValue), maximumFractionDigits)}`;
}


export function formatNumber(value, maximumFractionDigits = 0) {
    return new Intl.NumberFormat("en-US", {
        maximumFractionDigits,
    }).format(value || 0);
}


export function formatCompactNumber(value, maximumFractionDigits = 1) {
    return new Intl.NumberFormat("en-US", {
        notation: "compact",
        maximumFractionDigits,
    }).format(value || 0);
}


export function formatPrice(value) {
    return formatCurrency(value, 2);
}


export function formatMetric(value, maximumFractionDigits = 3) {
    return new Intl.NumberFormat("en-US", {
        minimumFractionDigits: maximumFractionDigits,
        maximumFractionDigits,
    }).format(value || 0);
}


export function formatPValue(value) {
    if (value > 0 && value < 0.001) {
        return "<0.001";
    }
    return formatMetric(value, 3);
}


export function formatDateLabel(value) {
    if (!value) {
        return "";
    }
    const [year, month] = value.split("-");
    return new Date(Number(year), Number(month) - 1).toLocaleDateString("en-US", {
        month: "short",
        year: "2-digit",
    });
}


export function formatCategory(value) {
    return String(value || "")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}


export function formatBoundaryHit(hit) {
    if (hit === "lower") {
        return "-15% boundary";
    }
    if (hit === "upper") {
        return "+15% boundary";
    }
    return "Interior";
}


export function cssTone(value) {
    if (value > 0) {
        return "positive";
    }
    if (value < 0) {
        return "negative";
    }
    return "neutral";
}
