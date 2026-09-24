"""Validate the authored map against the site's canonical publication data."""
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ResearchDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.works = tomllib.loads((ROOT / "data/research.toml").read_text(encoding="utf-8"))["works"]
        cls.papers = tomllib.loads((ROOT / "data/papers.toml").read_text(encoding="utf-8"))["papers"]
        taxonomy = tomllib.loads((ROOT / "data/research-taxonomy.en.toml").read_text(encoding="utf-8"))
        cls.assignments = {item["work_id"]: item for item in taxonomy["assignments"]}
        cls.terms = {item["id"]: item for item in taxonomy["terms"]}

    def test_every_publication_is_represented_exactly_once(self):
        references = [w["paper_url"] for w in self.works if "paper_url" in w]
        self.assertCountEqual(references, [p["paper_url"] for p in self.papers])
        self.assertEqual(len(references), len(set(references)))

    def test_deep_links_are_unique_and_all_nodes_fit_the_map(self):
        ids = [w["id"] for w in self.works]
        self.assertEqual(len(ids), len(set(ids)))
        for work in self.works:
            with self.subTest(work=work["id"]):
                self.assertRegex(work["id"], r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
                self.assertTrue(35 <= work["x"] <= 785)
                self.assertTrue(35 <= work["y"] <= 595)
                self.assertTrue(work["topics"])
                self.assertLessEqual(set(work["topics"]), {"systems", "embodied", "agents"})
                self.assertIn(work["role"], {"author", "collaborator", "section", "project"})
                self.assertGreaterEqual(work["contribution_weight"], 0.2)
                self.assertLessEqual(work["contribution_weight"], 1.0)
                self.assertGreaterEqual(work["node_radius"], 6)
                self.assertLessEqual(work["node_radius"], 13)
                for key in ("role_en", "role_zh", "summary_en", "summary_zh"):
                    self.assertTrue(work[key].strip())

    def test_rendered_work_assignments_match_the_english_taxonomy(self):
        for work in self.works:
            assignment = self.assignments[work["id"]]
            with self.subTest(work=work["id"]):
                self.assertEqual(work["term_ids"], assignment["primary"] + assignment["secondary"])
                self.assertEqual(work["primary_terms"], [self.terms[key]["label"] for key in assignment["primary"]])
                self.assertEqual(work["secondary_terms"], [self.terms[key]["label"] for key in assignment["secondary"]])

    def test_author_roles_are_supported_by_the_canonical_author_list(self):
        papers = {p["paper_url"]: p for p in self.papers}
        for work in self.works:
            if work["role"] == "author":
                authors = papers[work["paper_url"]]["authors"]
                prefix = authors.split("Xiangyu Li", 1)[0]
                self.assertTrue(not prefix or all("*" in author for author in prefix.split(",") if author.strip()), work["id"])
            elif work["role"] == "section":
                self.assertIn("Section Lead", papers[work["paper_url"]].get("notes", ""))

    def test_work_links_have_valid_schemes_and_bilingual_blog_targets(self):
        for entry in self.papers + self.works:
            for key, value in entry.items():
                if not key.endswith("_url"):
                    continue
                self.assertTrue(value.startswith(("https://", "/")), value)
                if key == "blog_url":
                    folder = ROOT / "content" / value.strip("/")
                    self.assertTrue((folder / "index.md").is_file(), value)
                    self.assertTrue((folder / "index.zh.md").is_file(), value)


if __name__ == "__main__":
    unittest.main()
