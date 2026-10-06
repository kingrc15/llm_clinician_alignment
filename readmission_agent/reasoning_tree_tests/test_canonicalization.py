import unittest

from reasoning_tree.app.canonicalization import canonicalize_step, get_canonicalizer


class CanonicalizationTests(unittest.TestCase):
    def test_lowercase_and_strip_punct(self):
        self.assertEqual(canonicalize_step("Elevated Troponin."), "elevated troponin")
        self.assertEqual(canonicalize_step("**Troponin** is elevated!"), "troponin is elevated")

    def test_removes_filler(self):
        self.assertEqual(canonicalize_step("I think the patient has heart failure."), "the patient has heart failure")

    def test_preserves_clinical_verbs(self):
        out = canonicalize_step("Elevated troponin suggests myocardial injury.")
        self.assertIn("suggests", out)
        self.assertIn("myocardial injury", out)

    def test_factory_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            get_canonicalizer("bogus")


if __name__ == "__main__":
    unittest.main()
