"""Core checks for scoring and SQL extraction. Run: python -m unittest discover tests"""

import json
import unittest
from pathlib import Path

from sqlbench.execution import execute, has_order_by, results_match
from sqlbench.prompts import extract_sql

ROOT = Path(__file__).resolve().parent.parent


class ExtractSQL(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(extract_sql("Here you go:\n```sql\nSELECT 1;\n```"), "SELECT 1;")

    def test_think_and_preamble(self):
        self.assertEqual(extract_sql("<think>select nothing</think>Sure! SELECT a FROM t; -- done"), "SELECT a FROM t;")

    def test_semicolon_in_string(self):
        self.assertEqual(extract_sql("SELECT 'a;b' FROM t; extra"), "SELECT 'a;b' FROM t;")

    def test_cte(self):
        self.assertTrue(extract_sql("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH"))


class Execution(unittest.TestCase):
    def test_read_only(self):
        for sql in ["DELETE FROM orders", "DROP TABLE orders", "SELECT 1; DROP TABLE orders",
                    "UPDATE products SET price = 0"]:
            self.assertFalse(execute(sql).ok, sql)
        self.assertEqual(execute("SELECT COUNT(*) FROM customers").rows, [(600,)])

    def test_replace_function_allowed(self):
        self.assertTrue(execute("SELECT REPLACE(name, ' ', '_') FROM products LIMIT 1").ok)

    def test_column_order_and_rounding(self):
        gold_sql = "SELECT brand, COUNT(*) FROM products GROUP BY brand"
        gold = execute(gold_sql)
        pred = execute("SELECT COUNT(*) AS n, brand FROM products GROUP BY brand ORDER BY n")
        self.assertTrue(results_match(gold, pred, gold_sql))

    def test_order_matters_only_with_order_by(self):
        gold_sql = "SELECT name FROM products ORDER BY price DESC LIMIT 3"
        wrong = execute("SELECT name FROM products ORDER BY price ASC LIMIT 3")
        self.assertFalse(results_match(execute(gold_sql), wrong, gold_sql))
        self.assertTrue(has_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a) ORDER BY a"))
        self.assertFalse(has_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)"))

    def test_extra_column_is_wrong(self):
        gold_sql = "SELECT COUNT(*) FROM orders"
        self.assertFalse(results_match(execute(gold_sql), execute("SELECT COUNT(*), 1 FROM orders"), gold_sql))

    def test_gold_set_is_valid(self):
        rows = [json.loads(line) for line in (ROOT / "data/gold_test.jsonl").read_text().splitlines() if line.strip()]
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        for r in rows:
            res = execute(r["sql"])
            self.assertTrue(res.ok and res.rows, r["id"])
            self.assertTrue(results_match(res, res, r["sql"]), r["id"])


if __name__ == "__main__":
    unittest.main()
