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
  let response = match;
  let ok = true;
  const timers = new Map();
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../clickless_ui/static/app.js'), 'utf8'), {
    document: { querySelector: get, createElement: tag => new Element(tag) },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => {
      if (unavailableStorage) throw new Error('Storage unavailable');
      storage.set(key, value);
    } },
    setTimeout: callback => { timers.set(1, callback); return 1; },
    clearTimeout: key => timers.delete(key),
    fetch: async () => ({ ok, json: async () => response }),
  });
  return { get, timers, storage,
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
  console.log('PASS: search, cart edits, removal undo, persistence, invalid data, and errors');
})().catch(error => { console.error(error); process.exitCode = 1; });
