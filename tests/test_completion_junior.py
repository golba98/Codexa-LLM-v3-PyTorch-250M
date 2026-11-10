from __future__ import annotations

import unittest

from scripts.datasets.mathematics.math_completion_workers import junior_2011 as module


class RecoveryTests(unittest.TestCase):
    def pages(self):
        pages = [""] * 218
        for number, title, printed in module.MODULES:
            index = printed + 5
            pages[index] += (
                f"{number} {title}\nCLASS WORK\nQuestion for {number}\nend of CLASS WORK\n"
            )
        return pages

    def test_shared_page_boundary_is_not_misattributed(self):
        pages = self.pages()
        pages[17] = "Previous module last line\n" + pages[17]
        rows, sections = module.recover(pages)
        first = "".join(x["original_text"] for x in rows if x["source_section"] == "1.1")
        second = "".join(x["original_text"] for x in rows if x["source_section"] == "1.2")
        self.assertIn("Previous module last line", first)
        self.assertNotIn("Previous module last line", second)
        self.assertEqual(len(sections), 25)
        self.assertEqual(len({x["record_id"] for x in rows}), len(rows))
        self.assertTrue(all(x["admission"] == "blocked" for x in rows))

    def test_missing_module_heading_fails_closed(self):
        pages = self.pages()
        pages[17] = ""
        with self.assertRaises(ValueError):
            module.recover(pages)

    def test_deterministic(self):
        self.assertEqual(module.recover(self.pages()), module.recover(self.pages()))


if __name__ == "__main__":
    unittest.main()
