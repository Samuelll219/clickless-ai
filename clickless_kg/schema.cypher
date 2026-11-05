// ClickLess AI knowledge graph schema (Neo4j 5.x)
//
// (:Product {code, name, brand, price, ingredients_text, traces})
//   -[:BELONGS_TO]->  (:Category {name, display, group, price_min, price_max})
//   -[:HAS_TAG]->     (:DietaryTag {name})
//   -[:CONTAINS]->    (:Allergen {name})
// (:Category)-[:SUBCATEGORY_OF]->(:Category {name, is_group: true})
//
// Products only ever BELONG_TO a leaf category. Group categories ("snacks",
// "dairy", ...) let a request such as "gluten-free snacks" match every leaf
// in that group. Statements are separated by ";" and run one at a time by load.py.

CREATE CONSTRAINT product_code IF NOT EXISTS FOR (p:Product) REQUIRE p.code IS UNIQUE;
CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT dietary_tag_name IF NOT EXISTS FOR (t:DietaryTag) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT allergen_name IF NOT EXISTS FOR (a:Allergen) REQUIRE a.name IS UNIQUE;
CREATE INDEX product_price IF NOT EXISTS FOR (p:Product) ON (p.price);
