"use strict";
// Add-ons, sandbox-store cart sync, interaction logging and order approval.
// Loaded after app.js and uses its globals (cart, changeCart, element, button, money).
(() => {
  const addonSection = document.querySelector("#addons");
  const addonList = document.querySelector("#addon-list");
  const confirmOrder = document.querySelector("#confirm-order");
  const orderStatus = document.querySelector("#order-status");
  const cartStorageKey = "clickless-ai-cart";
  const baseFetch = window.fetch.bind(window);
  let addonPending = false;

  const send = (url, method, body) => baseFetch(url, {
    method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then(r => r.json()).catch(() => null);
  const log = (event, detail) => send("/api/log", "POST", { event, detail });

  // ---- search: log the submission, then fetch add-ons for the top match
  window.fetch = async (url, options) => {
    const response = await baseFetch(url, options);
    if (url === "/api/search") {
      let request = null;
      try { request = JSON.parse(options.body).request; } catch { /* ignore */ }
      log("search_submit", { request });
      response.clone().json().then(result => {
        log("search_result", { status: result?.status, n: result?.products?.length ?? 0 });
        if (result?.status === "matched") showAddons(result); else hideAddons();
      }).catch(hideAddons);
    }
    return response;
  };

  function hideAddons() {
    if (addonSection) addonSection.hidden = true;
  }

  async function showAddons(result) {
    if (!addonSection) return;
    const top = result.products[0];
    const data = await send("/api/addons", "POST", { category: top.category, constraints: result.constraints });
    const items = data?.addons ?? [];
    addonList.replaceChildren();
    for (const addon of items) {
      const p = addon.product;
      const cents = Math.round(p.price * 100);
      const li = element("li");
      const head = element("div", undefined, "product-top");
      const title = element("div");
      title.append(element("h3", p.name), element("p", `${p.brand ?? ""} · ${addon.category.replaceAll("_", " ")}`, "meta"));
      head.append(title, element("span", `${money(cents)} each`, "price"));
      li.append(head, element("p", `Why: ${addon.explanation}`, "meta match-detail"));
      li.append(button("Add 1 to cart", () => {
        addonPending = true;
        const current = cart.get(p.code)?.quantity ?? 0;
        changeCart({ ...p, cents }, current + 1, `Added add-on ${p.name}.`);
        addonPending = false;
      }, `Add add-on ${p.name}`));
      addonList.append(li);
    }
    addonSection.hidden = items.length === 0;
    if (items.length) log("addon_shown", { categories: items.map(a => a.category) });
  }

  // ---- cart: mirror every saved change to the sandbox store and log it
  const readLines = raw => { try { return JSON.parse(raw ?? "[]"); } catch { return []; } };
  const baseSet = localStorage.setItem.bind(localStorage);
  localStorage.setItem = (key, value) => {
    const before = key === cartStorageKey ? readLines(localStorage.getItem(key)) : null;
    baseSet(key, value);
    if (key !== cartStorageKey) return;
    const after = readLines(value);
    const old = new Map(before.map(l => [l.code, l.quantity]));
    const now = new Map(after.map(l => [l.code, l.quantity]));
    if (!now.size && old.size) log("clear", null);
    for (const [code, q] of now) {
      const prev = old.get(code) ?? 0;
      if (q > prev) log(addonPending ? "addon_add" : (prev ? "quantity_change" : "add_to_cart"), { code, quantity: q });
      else if (q < prev) log("quantity_change", { code, quantity: q });
    }
    for (const code of old.keys()) if (!now.has(code) && now.size) log("remove", { code });
    send("/api/cart", "PUT", { lines: after.map(l => ({ code: l.code, quantity: l.quantity })) });
  };

  confirmOrder?.addEventListener("click", async () => {
    const lines = [...cart.values()].map(l => ({ code: l.code, quantity: l.quantity }));
    if (!lines.length) { orderStatus.textContent = "Your cart is empty."; return; }
    const saved = await send("/api/cart", "PUT", { lines });
    if (!saved || saved.error) {
      orderStatus.textContent = saved?.error ?? "The sandbox store is unavailable. Please retry.";
      return;
    }
    log("confirm_order", { items: saved.items, subtotal_cents: saved.subtotal_cents });
    orderStatus.textContent = `Order approved in the sandbox store: ${saved.items} items, ${money(saved.subtotal_cents)}.`;
  });

  // Start in sync with the browser's saved cart.
  send("/api/cart", "PUT", { lines: [...cart.values()].map(l => ({ code: l.code, quantity: l.quantity })) });
})();
