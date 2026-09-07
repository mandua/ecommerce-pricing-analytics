const API_BASE_URL = "";


async function requestJson(path, params = {}) {
    const url = new URL(`${API_BASE_URL}${path}`, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
            url.searchParams.set(key, value);
        }
    });

    const response = await fetch(url);

    if (!response.ok) {
        const fallbackMessage = `Request failed with status ${response.status}.`;
        let detail = fallbackMessage;
        try {
            const payload = await response.json();
            detail = payload.detail || fallbackMessage;
        } catch {
            detail = fallbackMessage;
        }
        throw new Error(detail);
    }

    return response.json();
}


export function fetchMetadata() {
    return requestJson("/api/metadata");
}


export function fetchExecutiveDashboard(filters) {
    return requestJson("/api/executive", filters);
}


export function fetchProductAnalytics(filters, productId) {
    return requestJson("/api/product-analytics", {
        ...filters,
        product_id: productId,
    });
}


export function fetchPricingLab(productId, selectedPrice) {
    return requestJson("/api/pricing-lab", {
        product_id: productId,
        selected_price: selectedPrice,
    });
}


export function fetchCustomerAnalytics(filters) {
    return requestJson("/api/customer-analytics", filters);
}


export function fetchExperimentation() {
    return requestJson("/api/experimentation");
}


export function fetchModelPerformance() {
    return requestJson("/api/model-performance");
}
