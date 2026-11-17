"use strict";

const form = document.querySelector("#search");
const requestInput = document.querySelector("#request");
const submit = form.querySelector("button");
const searchStatus = document.querySelector("#search-status");
const results = document.querySelector("#results");
const resultsTitle = document.querySelector("#results-title");
const constraintsList = document.querySelector("#constraints");
const productsList = document.querySelector("#products");
const cartItems = document.querySelector("#cart-items");
const cartStatus = document.querySelector("#cart-status");
const cartKey = "clickless-ai-cart";
const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const cart = loadCart();

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function button(label, action, accessibleLabel) {
  const node = element("button", label);
  node.type = "button";
  if (accessibleLabel) node.setAttribute("aria-label", accessibleLabel);
  node.addEventListener("click", action);
  return node;
}

function money(cents) {
  return dollars.format(cents / 100);
}

function displayName(value) {
  return value.replaceAll("_", " ");
}

function loadCart() {
  try {
    const lines = JSON.parse(localStorage.getItem(cartKey) ?? "[]");
    if (!Array.isArray(lines)) return new Map();
    return new Map(lines.filter(line =>
      line && typeof line.code === "string" && line.code.length > 0 &&
      typeof line.name === "string" && Number.isSafeInteger(line.cents) &&
      line.cents >= 0 && Number.isSafeInteger(line.quantity) && line.quantity > 0
    ).map(line => [line.code, line]));
  } catch {
    return new Map();
  }
}

function changeCart(product, quantity, message, focusAction) {
  if (!Number.isSafeInteger(quantity)) {
    cartStatus.textContent = "That quantity is too large.";
    return;
  }
  if (quantity > 0) {
    cart.set(product.code, { code: product.code, name: product.name, cents: product.cents, quantity });
  } else {
    cart.delete(product.code);
  }
  cartStatus.textContent = message;
  try {
    localStorage.setItem(cartKey, JSON.stringify([...cart.values()]));
  } catch {
    cartStatus.textContent += " Cart changes will last until this page closes.";
  }
  renderCart(product.code, focusAction);
}

function renderCart(focusCode, focusAction) {
  cartItems.replaceChildren();
  let subtotal = 0;
  let nextFocus;
  for (const line of cart.values()) {
    const item = element("li");
    const detail = element("div", undefined, "cart-line");
    const lineTotal = line.cents * line.quantity;
    subtotal += lineTotal;
    detail.append(element("h3", line.name), element("span", money(lineTotal), "price"));
    const controls = element("div", undefined, "controls");
    const decrease = button("−", () => changeCart(line, line.quantity - 1,
      line.quantity === 1 ? `Removed ${line.name}.` : `Updated ${line.name}.`, "decrease"),
    `Decrease quantity of ${line.name}`);
    const increase = button("+", () => changeCart(line, line.quantity + 1,
      `Updated ${line.name}.`, "increase"), `Increase quantity of ${line.name}`);
    const remove = button("Remove", () => changeCart(line, 0, `Removed ${line.name}.`, "remove"),
      `Remove ${line.name}`);
    controls.append(decrease, element("span", line.quantity), increase, remove);
    item.append(detail, element("p", `${money(line.cents)} each`, "meta"), controls);
    cartItems.append(item);
    if (line.code === focusCode) {
      nextFocus = focusAction === "decrease" ? decrease : increase;
    }
  }
  document.querySelector("#cart-empty").hidden = cart.size > 0;
  document.querySelector("#subtotal").textContent = money(subtotal);
  if (focusAction) (nextFocus ?? cartItems.querySelector("button") ?? requestInput).focus();
}

function renderResults(result) {
  const { constraints, products, quantity, status } = result;
  const fields = [
    ["Category", displayName(constraints.category)],
    ["Quantity", quantity],
    ["Dietary tags", constraints.dietary_tags.map(displayName).join(", ")],
    ["Exclude allergens", constraints.excluded_allergens.map(displayName).join(", ")],
    ["Max unit price", constraints.max_price === null ? "" : dollars.format(constraints.max_price)],
  ];

  constraintsList.replaceChildren();
  for (const [label, value] of fields) {
    if (value === "") continue;
    const item = element("div");
    item.append(element("dt", label), element("dd", value));
    constraintsList.append(item);
  }

  resultsTitle.textContent = status === "no_match" ? "No matching products" : "Matching products";
  searchStatus.textContent = status === "no_match"
    ? "No products satisfy all of the extracted constraints."
    : `${products.length} ${products.length === 1 ? "match" : "matches"} found.`;
  if (status === "matched") {
    for (const product of products) {
      const item = element("li");
      const top = element("div", undefined, "product-top");
      const bottom = element("div", undefined, "product-bottom");
      const title = element("div");
      const cents = Math.round(product.price * 100);
      title.append(element("h3", product.name));
      const brandAndCategory = [product.brand, displayName(product.category)].filter(Boolean);
      title.append(element("p", brandAndCategory.join(" · "), "meta"));
      top.append(title, element("span", `${money(cents)} each`, "price"));
      const add = button(`Add ${quantity} to cart`, () => {
        const currentQuantity = cart.get(product.code)?.quantity ?? 0;
        changeCart({ ...product, cents }, currentQuantity + quantity,
          `Added ${quantity} × ${product.name}.`);
      }, `Add ${quantity} to cart: ${product.name}`);
      bottom.append(element("span", `${money(cents * quantity)} for ${quantity}`, "meta"), add);
      item.append(top);
      if (product.dietary_tags.length) {
        item.append(element("p", product.dietary_tags.map(displayName).join(" · "), "meta"));
      }
      if (product.allergens.length) {
        item.append(element("p", `Contains ${product.allergens.map(displayName).join(", ")}`, "meta"));
      }
      item.append(bottom);
      productsList.append(item);
    }
  }
  results.hidden = false;
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submit.disabled) return;
  results.hidden = true;
  productsList.replaceChildren();
  searchStatus.classList.remove("error");
  const request = requestInput.value.trim();
  if (!request) {
    searchStatus.textContent = "Enter a shopping request.";
    requestInput.focus();
    return;
  }

  searchStatus.textContent = "Finding matching products…";
  submit.disabled = true;
  requestInput.readOnly = true;
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    });
    const result = await response.json();
    if (!response.ok) {
      searchStatus.classList.add("error");
      searchStatus.textContent = result.error;
      return;
    }
    renderResults(result);
  } catch {
    searchStatus.classList.add("error");
    searchStatus.textContent = "The search failed. Try again.";
  } finally {
    submit.disabled = false;
    requestInput.readOnly = false;
  }
});

renderCart();
