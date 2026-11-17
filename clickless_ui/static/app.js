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
const undoRemove = document.querySelector("#undo-remove");
let removedLine = null;
const cartKey = "clickless-ai-cart";
const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const cart = loadCart();

function validLine(line) {
  return line && typeof line.code === "string" && line.code.trim().length > 0 &&
    typeof line.name === "string" && line.name.trim().length > 0 &&
    Number.isSafeInteger(line.cents) && line.cents >= 0 &&
    Number.isSafeInteger(line.quantity) && line.quantity > 0 &&
    Number.isSafeInteger(line.cents * line.quantity);
}

function validCart(lines) {
  return lines.every(validLine) &&
    new Set(lines.map(line => line.code)).size === lines.length &&
    Number.isSafeInteger(lines.reduce((sum, line) => sum + line.cents * line.quantity, 0)) &&
    Number.isSafeInteger(lines.reduce((sum, line) => sum + line.quantity, 0));
}

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
    if (!Array.isArray(lines) || !validCart(lines)) {
      cartStatus.textContent = "The saved cart could not be restored. Please add your items again.";
      return new Map();
    }
    return new Map(lines.map(line => [line.code, line]));
  } catch {
    cartStatus.textContent = "The saved cart is unavailable. You can still shop in this tab.";
    return new Map();
  }
}

function changeCart(product, quantity, message, focusAction) {
  const proposed = [...cart.values()].filter(line => line.code !== product.code);
  if (quantity > 0) proposed.push({ ...product, quantity });
  if (!Number.isSafeInteger(quantity) || quantity < 0 || !validCart(proposed)) {
    cartStatus.textContent = "Enter a valid quantity within the supported total.";
    return false;
  }
  removedLine = quantity === 0 ? cart.get(product.code) : null;
  undoRemove.hidden = !removedLine;
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
  return true;
}

undoRemove.addEventListener("click", () => {
  if (!removedLine) return;
  const line = removedLine;
  changeCart(line, line.quantity, `Restored ${line.name}.`, "increase");
});

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
    const quantityInput = element("input", undefined, "quantity");
    quantityInput.type = "number";
    quantityInput.min = "1";
    quantityInput.step = "1";
    quantityInput.value = line.quantity;
    quantityInput.setAttribute("aria-label", `Quantity of ${line.name}`);
    quantityInput.addEventListener("change", () => {
      const quantity = Number(quantityInput.value);
      if (quantity < 1 || !changeCart(line, quantity, `Updated ${line.name}.`, "quantity")) {
        quantityInput.value = line.quantity;
        cartStatus.textContent = "Enter a positive whole-number quantity within the supported total.";
      }
    });
    controls.append(decrease, quantityInput, increase, remove);
    item.append(detail, element("p", `${money(line.cents)} each · ${line.code}`, "meta"), controls);
    cartItems.append(item);
    if (line.code === focusCode) {
      nextFocus = focusAction === "quantity" ? quantityInput : focusAction === "decrease" ? decrease : increase;
    }
  }
  document.querySelector("#cart-empty").hidden = cart.size > 0;
  document.querySelector("#subtotal").textContent = money(subtotal);
  const count = [...cart.values()].reduce((sum, line) => sum + line.quantity, 0);
  document.querySelector("#cart-count").textContent = `${count} ${count === 1 ? "item" : "items"}`;
  if (focusAction) (nextFocus ?? cartItems.querySelector("button") ?? requestInput).focus();
}

function renderResults(result) {
  const { constraints, products, quantity, status } = result ?? {};
  const stringList = value => Array.isArray(value) && value.every(item => typeof item === "string");
  if (!constraints || typeof constraints.category !== "string" ||
      !stringList(constraints.dietary_tags) || !stringList(constraints.excluded_allergens) ||
      !(constraints.max_price === null || (Number.isFinite(constraints.max_price) && constraints.max_price >= 0)) ||
      !Number.isSafeInteger(quantity) || quantity < 1 || !Array.isArray(products) ||
      !["matched", "no_match"].includes(status) || (status === "matched") !== (products.length > 0) ||
      !products.every(product => product && typeof product.category === "string" &&
        (product.brand == null || typeof product.brand === "string") &&
        stringList(product.dietary_tags) && stringList(product.allergens) &&
        Number.isFinite(product.price) && product.price >= 0 &&
        validLine({ ...product, cents: Math.round(product.price * 100), quantity }))) {
    throw new Error("The search returned incomplete product information. Please try again.");
  }
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
    ? "No products satisfy all of these constraints. Edit your request to search again."
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
      } else {
        item.append(element("p", "No allergens recorded in this catalog.", "meta"));
      }
      item.append(element("p", `Barcode ${product.code}`, "meta"));
      if (constraints.excluded_allergens.length) {
        item.append(element("p", `Excluded allergens absent from catalog record: ${constraints.excluded_allergens.map(displayName).join(", ")}`, "meta match-detail"));
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
  if (!request || request.length > 500) {
    searchStatus.textContent = "Enter a shopping request between 1 and 500 characters.";
    requestInput.focus();
    return;
  }

  searchStatus.textContent = "Finding matching products…";
  submit.disabled = true;
  submit.textContent = "Searching…";
  form.setAttribute("aria-busy", "true");
  requestInput.readOnly = true;
  const slowSearch = setTimeout(() => {
    searchStatus.textContent = "Still searching. The first request may take longer while the model loads.";
  }, 15000);
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request }),
    });
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("The search service returned an unreadable response. Please try again.");
    }
    if (!response.ok) {
      searchStatus.classList.add("error");
      searchStatus.textContent = result?.error || "The search service is unavailable. Please try again.";
      return;
    }
    renderResults(result);
  } catch (error) {
    searchStatus.classList.add("error");
    searchStatus.textContent = error instanceof TypeError
      ? "Unable to reach the search service. Check your connection and try again."
      : error.message;
  } finally {
    clearTimeout(slowSearch);
    form.setAttribute("aria-busy", "false");
    submit.disabled = false;
    submit.textContent = "Find products";
    requestInput.readOnly = false;
  }
});

renderCart();
