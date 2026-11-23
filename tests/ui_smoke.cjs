// Run with node tests/ui_smoke.cjs. This uses a small DOM double, not a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.classList = new Set();
    this.classList.remove = this.classList.delete;
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children = []; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  focus() {}
  contains(node) { return this.children.some(child => child === node || child.contains(node)); }
  querySelector(tag) {
    for (const child of this.children) {
      if (child.tag === tag) return child;
      const found = child.querySelector(tag);
      if (found) return found;
    }
  }
}

const product = { code: '001234', name: '<b>Protein bar</b>', price: 2.15,
  category: 'snack_bars', brand: 'Demo', dietary_tags: [], allergens: [] };
const match = { status: 'matched', quantity: 2, products: [product], constraints: {
  category: 'snack_bars', dietary_tags: [], excluded_allergens: [], max_price: 5,
} };

function page(storage = new Map(), unavailableStorage = false) {
  const nodes = new Map();
  const get = selector => {
    if (!nodes.has(selector)) nodes.set(selector, new Element(selector));
    return nodes.get(selector);
  };
  get('#search').append(new Element('button'));
  get('#request').value = 'two bars';
  get('#sort-products').value = 'original';
  const example = get('#example');
  example.dataset = { request: 'Find 1 vegan plant milk under $6 each.' };
  let response = match;
  let ok = true;
  let pending = false;
  let calls = 0;
  let timerId = 0;
  const timers = new Map();
  const events = {};
  const localStorage = { getItem: key => storage.get(key), setItem: (key, value) => {
    if (unavailableStorage) throw new Error('Storage unavailable');
    storage.set(key, value);
  } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../clickless_ui/static/app.js'), 'utf8'), {
    document: { querySelector: get, querySelectorAll: () => [example], createElement: tag => new Element(tag) },
    window: { addEventListener: (name, listener) => { events[name] = listener; } },
    localStorage, AbortController,
    setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
    clearTimeout: key => timers.delete(key),
    fetch: async (url, options) => {
      calls++;
      if (pending) await new Promise((resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('Aborted')), { once: true });
      });
      return { ok, json: async () => response };
    },
  });
  get('#search').requestSubmit = () => get('#search').listeners.submit({ preventDefault() {} });
  return { get, timers, storage,
    calls: () => calls,
    hold: value => { pending = value; },
    storageEvent: (key = 'clickless-ai-cart') => events.storage({ key, storageArea: localStorage }),
    search: () => get('#search').listeners.submit({ preventDefault() {} }),
    reply: (value, success = true) => { response = value; ok = success; },
    add: () => get('#products').children[0].querySelector('button').listeners.click(),
    controls: () => get('#cart-items').children[0].children[2].children,
  };
}

(async () => {
  const p = page();
  await p.search();
  assert.equal(p.get('#products').children[0].querySelector('h3').textContent, product.name);
  p.add(); p.add();
  assert.equal(p.get('#subtotal').textContent, '$8.60');
  assert.equal(JSON.parse(p.storage.get('clickless-ai-cart'))[0].code, '001234');
  p.controls()[0].listeners.click();
  assert.equal(p.get('#subtotal').textContent, '$6.45');
  p.controls()[2].listeners.click();
  const input = p.controls()[1];
  input.value = '7'; input.listeners.change();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  const invalid = p.controls()[1];
  invalid.value = '1.5'; invalid.listeners.change();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  p.controls()[3].listeners.click();
  assert.equal(p.get('#subtotal').textContent, '$0.00');
  p.get('#undo-remove').listeners.click();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  assert.equal(page(p.storage).get('#subtotal').textContent, '$15.05');
  p.reply({ ...match, status: 'no_match', products: [] });
  await p.search();
  assert.equal(p.get('#products').children.length, 0);
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  p.reply({ error: 'Database unavailable' }, false);
  await p.search();
  assert.equal(p.get('#search-status').textContent, 'Database unavailable');
  for (const invalidProduct of [{ ...product, code: 1234 }, { ...product, price: -1 },
    { ...product, price: Infinity }, { ...product, brand: {} }]) {
    p.reply({ ...match, products: [invalidProduct] });
    await p.search();
    assert.equal(p.get('#products').children.length, 0);
    assert.equal(p.get('#results').hidden, true);
  }
  p.reply(null);
  await p.search();
  assert.match(p.get('#search-status').textContent, /incomplete product information/);
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  assert.equal(p.timers.size, 0);
  assert.equal(p.get('#search')['aria-busy'], 'false');
  const corrupt = page(new Map([['clickless-ai-cart', 'broken JSON']]));
  assert.equal(corrupt.get('#subtotal').textContent, '$0.00');
  assert.ok(corrupt.get('#cart-status').textContent);
  const savedLine = { code: '001234', name: 'Bar', cents: 215, quantity: 2 };
  const duplicate = page(new Map([['clickless-ai-cart', JSON.stringify([savedLine, savedLine])]]));
  assert.equal(duplicate.get('#subtotal').textContent, '$0.00');
  assert.match(duplicate.get('#cart-status').textContent, /could not be restored/);
  const noStorage = page(new Map(), true);
  await noStorage.search(); noStorage.add();
  assert.equal(noStorage.get('#subtotal').textContent, '$4.30');
  assert.match(noStorage.get('#cart-status').textContent, /until this page closes/);
  p.get('#clear-cart').listeners.click();
  assert.equal(p.get('#subtotal').textContent, '$0.00');
  assert.equal(p.get('#clear-cart').disabled, true);
  p.get('#undo-remove').listeners.click();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  assert.match(p.get('#mobile-cart').textContent, /7 items.*\$15\.05/);

  // Two pages sharing storage: updates, remote clears, and invalid updates.
  const other = page(p.storage);
  other.get('#clear-cart').listeners.click();
  p.storageEvent();
  assert.equal(p.get('#subtotal').textContent, '$0.00');
  assert.equal(p.get('#undo-remove').hidden, true);
  other.get('#undo-remove').listeners.click();
  p.storageEvent();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  p.storage.set('clickless-ai-cart', 'broken');
  p.storageEvent();
  assert.equal(p.get('#subtotal').textContent, '$15.05');
  assert.match(p.get('#cart-status').textContent, /unchanged/);
  p.storage.delete('clickless-ai-cart');
  p.storageEvent(null);
  assert.equal(p.get('#subtotal').textContent, '$0.00');

  const sorted = page();
  const cheap = { ...product, code: '000007', name: 'Apple bar', price: 1 };
  sorted.reply({ ...match, products: [product, cheap] });
  await sorted.search();
  sorted.get('#sort-products').value = 'price-asc';
  sorted.get('#sort-products').listeners.change();
  assert.equal(sorted.get('#products').children[0].querySelector('h3').textContent, 'Apple bar');
  assert.equal(sorted.get('#products').children.length, 2);
  sorted.add();
  assert.equal(sorted.get('#subtotal').textContent, '$2.00');
  sorted.get('#sort-products').value = 'price-desc';
  sorted.get('#sort-products').listeners.change();
  assert.equal(sorted.get('#products').children[0].querySelector('h3').textContent, product.name);
  sorted.get('#sort-products').value = 'name';
  sorted.get('#sort-products').listeners.change();
  assert.equal(sorted.get('#products').children.length, 2);
  assert.deepEqual(sorted.get('#products').children.map(item => item.querySelector('h3').textContent),
    [product.name, cheap.name].sort((a, b) => a.localeCompare(b)));
  sorted.get('#example').listeners.click();
  assert.equal(sorted.get('#request').value, sorted.get('#example').dataset.request);

  const slow = page();
  await slow.search(); slow.add();
  slow.hold(true);
  let running = slow.search();
  await slow.search();
  assert.equal(slow.calls(), 2, 'Duplicate submit must not send another request');
  assert.equal(slow.get('#cancel-search').hidden, false);
  [...slow.timers.values()].find(timer => timer.delay === 15000).callback();
  assert.match(slow.get('#search-status').textContent, /Still searching/);
  slow.get('#cancel-search').listeners.click();
  await running;
  assert.match(slow.get('#search-status').textContent, /cancelled/);
  assert.equal(slow.get('#subtotal').textContent, '$4.30');
  assert.equal(slow.get('#retry-search').hidden, false);
  assert.equal(slow.get('#search').querySelector('button').disabled, false);
  assert.equal(slow.timers.size, 0);
  running = slow.search();
  [...slow.timers.values()].find(timer => timer.delay === 180000).callback();
  await running;
  assert.match(slow.get('#search-status').textContent, /timed out/);
  assert.equal(slow.get('#subtotal').textContent, '$4.30');
  assert.equal(slow.timers.size, 0);
  slow.hold(false);
  await slow.get('#retry-search').listeners.click();
  assert.equal(slow.get('#products').children.length, 1);
  assert.equal(slow.get('#retry-search').hidden, true);
  console.log('PASS: cart edits/undo, storage sync, sorting, examples, cancellation, timeout, retry, and validation');
})().catch(error => { console.error(error); process.exitCode = 1; });
