import {
    fetchCustomerAnalytics,
    fetchExecutiveDashboard,
    fetchExperimentation,
    fetchMetadata,
    fetchModelPerformance,
    fetchPricingLab,
    fetchProductAnalytics,
} from "./api.js";
import {
    populateSelect,
    renderCustomerAnalytics,
    renderExecutiveDashboard,
    renderExperimentation,
    renderModelPerformance,
    renderPricingLab,
    renderProductAnalytics,
    setStatus,
} from "./components.js";
import { formatCategory, formatPrice } from "./formatters.js";


const state = {
    metadata: null,
    selectedProductId: null,
    pricingProductId: null,
    pricingPrice: null,
    pricingTimer: null,
    activeDatePicker: null,
    calendarMonths: {
        start: null,
        end: null,
    },
};


const elements = {
    status: document.getElementById("app-status"),
    content: document.getElementById("app-shell"),
    disclaimer: document.getElementById("project-disclaimer"),
    navButtons: [...document.querySelectorAll("[data-view-target]")],
    views: [...document.querySelectorAll("[data-view]")],
    startDate: document.getElementById("filter-start-date"),
    endDate: document.getElementById("filter-end-date"),
    startDateButton: document.getElementById("filter-start-date-button"),
    endDateButton: document.getElementById("filter-end-date-button"),
    startDateValue: document.getElementById("filter-start-date-value"),
    endDateValue: document.getElementById("filter-end-date-value"),
    startDateCalendar: document.getElementById("filter-start-date-calendar"),
    endDateCalendar: document.getElementById("filter-end-date-calendar"),
    region: document.getElementById("filter-region"),
    category: document.getElementById("filter-category"),
    filterNote: document.getElementById("filter-scope-note"),
    executive: document.getElementById("executive-content"),
    product: document.getElementById("product-content"),
    productSelector: document.getElementById("product-selector"),
    pricing: document.getElementById("pricing-content"),
    pricingProductSelector: document.getElementById("pricing-product-selector"),
    pricingSlider: document.getElementById("pricing-price-slider"),
    pricingValue: document.getElementById("pricing-price-value"),
    pricingRange: document.getElementById("pricing-range-label"),
    customer: document.getElementById("customer-content"),
    experimentation: document.getElementById("experimentation-content"),
    model: document.getElementById("model-content"),
};

const datePickers = {
    start: {
        input: elements.startDate,
        button: elements.startDateButton,
        value: elements.startDateValue,
        calendar: elements.startDateCalendar,
        label: "Start Date",
    },
    end: {
        input: elements.endDate,
        button: elements.endDateButton,
        value: elements.endDateValue,
        calendar: elements.endDateCalendar,
        label: "End Date",
    },
};

const DAY_NAMES = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
const MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const BRAND_DISCLAIMER =
    "PriceLabs is an independent portfolio project using synthetic Gymshark-style data. It is not affiliated with or endorsed by Gymshark.";


document.addEventListener("DOMContentLoaded", initDashboard);


async function initDashboard() {
    setStatus(elements.status, "", "Loading PriceLabs analytics...");

    try {
        state.metadata = await fetchMetadata();
        configureFilters(state.metadata);
        bindEvents();

        await Promise.all([
            refreshFilteredViews(),
            refreshPricingLab(),
            refreshExperimentation(),
            refreshModelPerformance(),
        ]);

        elements.disclaimer.textContent = BRAND_DISCLAIMER;
        elements.content.classList.remove("is-hidden");
        setStatus(elements.status, "success", "PriceLabs synthetic analytics loaded.");
        activateView("executive");
    } catch (error) {
        setStatus(
            elements.status,
            "error",
            `Unable to load dashboard data: ${error.message}`
        );
    }
}


function configureFilters(metadata) {
    const { min, max } = metadata.date_range;
    elements.startDate.min = min;
    elements.startDate.max = max;
    elements.startDate.value = min;
    elements.endDate.min = min;
    elements.endDate.max = max;
    elements.endDate.value = max;
    state.calendarMonths.start = monthKeyFromIso(min);
    state.calendarMonths.end = monthKeyFromIso(max);

    populateSelect(elements.region, metadata.regions, "all", {
        includeAll: true,
        allLabel: "All regions",
    });
    populateSelect(elements.category, metadata.categories, "all", {
        includeAll: true,
        allLabel: "All categories",
        labelKey: "label",
        valueKey: "value",
    });
    elements.category.innerHTML =
        `<option value="all">All categories</option>` +
        metadata.categories
            .map(
                (category) =>
                    `<option value="${category}">${formatCategory(category)}</option>`
            )
            .join("");

    normalizeDateFilters();
    renderDatePickers();
}


function bindEvents() {
    elements.navButtons.forEach((button) => {
        button.addEventListener("click", () => activateView(button.dataset.viewTarget));
    });

    Object.entries(datePickers).forEach(([role, picker]) => {
        picker.button.addEventListener("click", (event) => {
            event.stopPropagation();
            toggleDatePicker(role);
        });

        picker.button.addEventListener("keydown", (event) => {
            if (["Enter", " ", "ArrowDown"].includes(event.key)) {
                event.preventDefault();
                openDatePicker(role);
                focusCalendarDate(role, picker.input.value);
            }
        });

        picker.calendar.addEventListener("click", (event) => {
            event.stopPropagation();
            handleCalendarClick(event, role);
        });

        picker.calendar.addEventListener("keydown", (event) => {
            handleCalendarKeydown(event, role);
        });
    });

    document.addEventListener("click", (event) => {
        if (!event.target.closest(".date-picker-field")) {
            closeDatePickers();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeDatePickers();
        }
    });

    [elements.region, elements.category].forEach(
        (element) => element.addEventListener("change", refreshFilteredViews)
    );

    elements.productSelector.addEventListener("change", () => {
        state.selectedProductId = elements.productSelector.value;
        refreshProductAnalytics();
    });

    elements.pricingProductSelector.addEventListener("change", () => {
        state.pricingProductId = elements.pricingProductSelector.value;
        state.pricingPrice = null;
        refreshPricingLab();
    });

    elements.pricingSlider.addEventListener("input", () => {
        state.pricingPrice = Number(elements.pricingSlider.value);
        elements.pricingValue.textContent = formatPrice(state.pricingPrice);
        window.clearTimeout(state.pricingTimer);
        state.pricingTimer = window.setTimeout(refreshPricingLab, 80);
    });
}


async function refreshFilteredViews() {
    await Promise.all([
        refreshExecutiveDashboard(),
        refreshProductAnalytics(),
        refreshCustomerAnalytics(),
    ]);
}


async function refreshExecutiveDashboard() {
    const payload = await fetchExecutiveDashboard(currentFilters());
    renderExecutiveDashboard(elements.executive, payload);
}


async function refreshProductAnalytics() {
    const payload = await fetchProductAnalytics(
        productFilters(),
        state.selectedProductId
    );
    if (payload.products.length) {
        const selectedId = payload.selected_product.product_id;
        state.selectedProductId = selectedId;
        populateProductSelector(elements.productSelector, payload.products, selectedId);
    }
    renderProductAnalytics(elements.product, payload);
}


async function refreshPricingLab() {
    const payload = await fetchPricingLab(state.pricingProductId, state.pricingPrice);
    state.pricingProductId = payload.product.product_id;
    state.pricingPrice = payload.selected_scenario.proposed_price;

    populateProductSelector(
        elements.pricingProductSelector,
        payload.products,
        state.pricingProductId
    );
    configurePricingSlider(payload.supported_range, state.pricingPrice);
    renderPricingLab(elements.pricing, payload);
}


async function refreshCustomerAnalytics() {
    const payload = await fetchCustomerAnalytics(currentFilters());
    renderCustomerAnalytics(elements.customer, payload);
}


async function refreshExperimentation() {
    const payload = await fetchExperimentation();
    renderExperimentation(elements.experimentation, payload);
}


async function refreshModelPerformance() {
    const payload = await fetchModelPerformance();
    renderModelPerformance(elements.model, payload);
}


function configurePricingSlider(range, selectedPrice) {
    elements.pricingSlider.min = range.min;
    elements.pricingSlider.max = range.max;
    elements.pricingSlider.step = range.step;
    elements.pricingSlider.value = selectedPrice;
    elements.pricingValue.textContent = formatPrice(selectedPrice);
    elements.pricingRange.textContent = `${formatPrice(range.min)} to ${formatPrice(range.max)} supported range`;
}


function populateProductSelector(element, products, selectedId) {
    element.innerHTML = products
        .map(
            (product) => `
                <option value="${product.product_id}">
                    ${product.product_name} - ${formatCategory(product.product_category)}
                </option>
            `
        )
        .join("");
    element.value = selectedId;
}


function currentFilters() {
    normalizeDateFilters();

    return {
        start_date: elements.startDate.value,
        end_date: elements.endDate.value,
        region: elements.region.value,
        category: elements.category.value,
    };
}


function productFilters() {
    return currentFilters();
}


function normalizeDateFilters() {
    const bounds = state.metadata?.date_range;
    if (!bounds) {
        return;
    }

    elements.startDate.value = clampDateValue(
        elements.startDate.value,
        bounds.min,
        bounds.max,
        bounds.min
    );
    elements.endDate.value = clampDateValue(
        elements.endDate.value,
        bounds.min,
        bounds.max,
        bounds.max
    );

    if (elements.startDate.value > elements.endDate.value) {
        elements.endDate.value = elements.startDate.value;
    }

    Object.keys(datePickers).forEach((role) => {
        const picker = datePickers[role];
        state.calendarMonths[role] = constrainVisibleMonth(
            role,
            state.calendarMonths[role] || monthKeyFromIso(picker.input.value)
        );
    });

    renderDatePickers();
}


function clampDateValue(value, min, max, fallback) {
    if (!value) {
        return fallback;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}


function renderDatePickers() {
    Object.keys(datePickers).forEach((role) => {
        datePickers[role].value.textContent = formatDisplayDate(
            datePickers[role].input.value
        );
        renderCalendar(role);
    });
}


function toggleDatePicker(role) {
    if (state.activeDatePicker === role) {
        closeDatePicker(role);
        return;
    }

    openDatePicker(role);
    focusCalendarDate(role, datePickers[role].input.value);
}


function openDatePicker(role) {
    Object.keys(datePickers).forEach((pickerRole) => {
        if (pickerRole !== role) {
            closeDatePicker(pickerRole);
        }
    });

    state.activeDatePicker = role;
    state.calendarMonths[role] = constrainVisibleMonth(
        role,
        state.calendarMonths[role] || monthKeyFromIso(datePickers[role].input.value)
    );
    renderCalendar(role);
    datePickers[role].calendar.hidden = false;
    datePickers[role].button.setAttribute("aria-expanded", "true");
}


function closeDatePickers() {
    Object.keys(datePickers).forEach(closeDatePicker);
}


function closeDatePicker(role) {
    datePickers[role].calendar.hidden = true;
    datePickers[role].button.setAttribute("aria-expanded", "false");
    if (state.activeDatePicker === role) {
        state.activeDatePicker = null;
    }
}


function handleCalendarClick(event, role) {
    const actionButton = event.target.closest("[data-calendar-action]");
    if (actionButton) {
        const direction = actionButton.dataset.calendarAction === "next" ? 1 : -1;
        moveCalendarMonth(role, direction);
        focusFirstSelectableCalendarDay(role);
        return;
    }

    const dayButton = event.target.closest("[data-calendar-date]");
    if (!dayButton || dayButton.disabled) {
        return;
    }

    selectDate(role, dayButton.dataset.calendarDate);
}


function handleCalendarKeydown(event, role) {
    if (event.key === "Escape") {
        event.preventDefault();
        closeDatePicker(role);
        datePickers[role].button.focus();
        return;
    }

    const dayButton = event.target.closest("[data-calendar-date]");
    if (!dayButton) {
        return;
    }

    const currentDate = parseIsoDate(dayButton.dataset.calendarDate);
    let nextDate = null;

    if (event.key === "ArrowLeft") {
        nextDate = addDays(currentDate, -1);
    } else if (event.key === "ArrowRight") {
        nextDate = addDays(currentDate, 1);
    } else if (event.key === "ArrowUp") {
        nextDate = addDays(currentDate, -7);
    } else if (event.key === "ArrowDown") {
        nextDate = addDays(currentDate, 7);
    } else if (event.key === "Home") {
        nextDate = addDays(currentDate, -currentDate.getDay());
    } else if (event.key === "End") {
        nextDate = addDays(currentDate, 6 - currentDate.getDay());
    } else if (event.key === "PageUp") {
        nextDate = addMonths(currentDate, -1);
    } else if (event.key === "PageDown") {
        nextDate = addMonths(currentDate, 1);
    }

    if (!nextDate) {
        return;
    }

    event.preventDefault();
    const nextIso = clampDateForPicker(role, toIsoDate(nextDate));
    state.calendarMonths[role] = monthKeyFromIso(nextIso);
    renderCalendar(role);
    focusCalendarDate(role, nextIso);
}


function selectDate(role, isoDate) {
    if (!isSelectableDate(role, isoDate)) {
        return;
    }

    datePickers[role].input.value = isoDate;
    state.calendarMonths[role] = monthKeyFromIso(isoDate);
    normalizeDateFilters();
    closeDatePicker(role);
    refreshFilteredViews();
}


function renderCalendar(role) {
    const picker = datePickers[role];
    const monthKey = constrainVisibleMonth(
        role,
        state.calendarMonths[role] || monthKeyFromIso(picker.input.value)
    );
    state.calendarMonths[role] = monthKey;

    const [year, month] = monthKey.split("-").map(Number);
    const firstOfMonth = new Date(year, month - 1, 1);
    const gridStart = addDays(firstOfMonth, -firstOfMonth.getDay());
    const titleId = `${role}-date-calendar-title`;
    const canGoPrevious = canNavigateMonth(role, -1);
    const canGoNext = canNavigateMonth(role, 1);

    const dayButtons = Array.from({ length: 42 }, (_, index) => {
        const day = addDays(gridStart, index);
        const isoDate = toIsoDate(day);
        const outsideMonth = day.getMonth() !== firstOfMonth.getMonth();
        const selectable = !outsideMonth && isSelectableDate(role, isoDate);
        const selected = isoDate === picker.input.value;

        return `
            <button
                class="calendar-day ${outsideMonth ? "outside-month" : ""} ${selected ? "selected" : ""}"
                type="button"
                data-calendar-date="${isoDate}"
                aria-selected="${selected ? "true" : "false"}"
                tabindex="${selected ? "0" : "-1"}"
                ${selectable ? "" : "disabled"}
            >
                ${day.getDate()}
            </button>
        `;
    }).join("");

    picker.calendar.innerHTML = `
        <div class="calendar-header">
            <button
                class="calendar-nav"
                type="button"
                data-calendar-action="previous"
                aria-label="Previous month"
                ${canGoPrevious ? "" : "disabled"}
            >
                &lt;
            </button>
            <strong id="${titleId}">${MONTH_NAMES[month - 1]} ${year}</strong>
            <button
                class="calendar-nav"
                type="button"
                data-calendar-action="next"
                aria-label="Next month"
                ${canGoNext ? "" : "disabled"}
            >
                &gt;
            </button>
        </div>
        <div class="calendar-weekdays" aria-hidden="true">
            ${DAY_NAMES.map((dayName) => `<span>${dayName}</span>`).join("")}
        </div>
        <div class="calendar-grid" role="grid" aria-labelledby="${titleId}">
            ${dayButtons}
        </div>
        <p class="calendar-range-note">${formatDisplayDate(pickerBounds(role).min)} to ${formatDisplayDate(pickerBounds(role).max)}</p>
    `;
}


function moveCalendarMonth(role, direction) {
    const currentMonth = state.calendarMonths[role];
    const nextMonth = addMonthsToKey(currentMonth, direction);
    state.calendarMonths[role] = constrainVisibleMonth(role, nextMonth);
    renderCalendar(role);
}


function focusCalendarDate(role, isoDate) {
    window.requestAnimationFrame(() => {
        const picker = datePickers[role];
        const dateButton =
            picker.calendar.querySelector(
                `[data-calendar-date="${isoDate}"]:not(:disabled)`
            ) || picker.calendar.querySelector("[data-calendar-date]:not(:disabled)");
        dateButton?.focus();
    });
}


function focusFirstSelectableCalendarDay(role) {
    window.requestAnimationFrame(() => {
        datePickers[role].calendar
            .querySelector("[data-calendar-date]:not(:disabled)")
            ?.focus();
    });
}


function pickerBounds(role) {
    const bounds = state.metadata.date_range;
    let min = bounds.min;
    let max = bounds.max;

    if (role === "start" && elements.endDate.value < max) {
        max = elements.endDate.value;
    }

    if (role === "end" && elements.startDate.value > min) {
        min = elements.startDate.value;
    }

    return { min, max };
}


function isSelectableDate(role, isoDate) {
    const bounds = pickerBounds(role);
    return isoDate >= bounds.min && isoDate <= bounds.max;
}


function clampDateForPicker(role, isoDate) {
    const bounds = pickerBounds(role);
    return clampDateValue(isoDate, bounds.min, bounds.max, datePickers[role].input.value);
}


function constrainVisibleMonth(role, monthKey) {
    const bounds = pickerBounds(role);
    const monthStart = `${monthKey}-01`;
    const monthEnd = toIsoDate(lastDayOfMonth(monthKey));

    if (monthEnd < bounds.min) {
        return monthKeyFromIso(bounds.min);
    }
    if (monthStart > bounds.max) {
        return monthKeyFromIso(bounds.max);
    }
    return monthKey;
}


function canNavigateMonth(role, direction) {
    const targetMonth = addMonthsToKey(state.calendarMonths[role], direction);
    const bounds = pickerBounds(role);
    return (
        `${targetMonth}-01` <= bounds.max &&
        toIsoDate(lastDayOfMonth(targetMonth)) >= bounds.min
    );
}


function monthKeyFromIso(isoDate) {
    return isoDate.slice(0, 7);
}


function parseIsoDate(isoDate) {
    const [year, month, day] = isoDate.split("-").map(Number);
    return new Date(year, month - 1, day);
}


function toIsoDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
}


function addDays(date, days) {
    const nextDate = new Date(date);
    nextDate.setDate(nextDate.getDate() + days);
    return nextDate;
}


function addMonths(date, months) {
    const nextDate = new Date(date);
    nextDate.setMonth(nextDate.getMonth() + months);
    return nextDate;
}


function addMonthsToKey(monthKey, months) {
    const [year, month] = monthKey.split("-").map(Number);
    return toIsoDate(new Date(year, month - 1 + months, 1)).slice(0, 7);
}


function lastDayOfMonth(monthKey) {
    const [year, month] = monthKey.split("-").map(Number);
    return new Date(year, month, 0);
}


function formatDisplayDate(isoDate) {
    const date = parseIsoDate(isoDate);
    return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
    });
}


function activateView(viewName) {
    const filterNotes = {
        executive:
            "Filters apply to sales, orders, churn scope, and date/category funnel data. Region does not apply to product-day funnel events.",
        products:
            "Date, region, and category filters apply to transaction revenue and margin. Stockouts, views, current price, and elasticity are product-level signals and are not region-specific.",
        pricing:
            "Pricing Lab simulations use the full product history and optimizer output. Global filters are not applied to modeled price scenarios.",
        customers:
            "Filters define the customer scope through matching transactions. Churn is a customer-level outcome; cohort cells show repeat-purchase activity.",
        experimentation:
            "Experimentation results use the full regenerated customer dataset. Global filters are not applied to A/B or causal summaries.",
        models:
            "Model metrics use the regenerated held-out test split. Global filters are not applied to model-performance summaries.",
    };

    elements.filterNote.textContent = filterNotes[viewName] || filterNotes.executive;

    elements.navButtons.forEach((button) => {
        const isActive = button.dataset.viewTarget === viewName;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-current", isActive ? "page" : "false");
    });

    elements.views.forEach((view) => {
        view.classList.toggle("active", view.dataset.view === viewName);
    });
}
