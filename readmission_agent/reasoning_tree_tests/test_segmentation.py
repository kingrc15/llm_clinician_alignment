import unittest

from reasoning_tree.app.segmentation import segment_response, segment_text


class SegmentationTests(unittest.TestCase):
    def test_segment_text_bullets_and_sentences(self):
        text = (
            "- Patient has elevated troponin.\n"
            "- ECG shows ST elevation. Likely STEMI.\n"
            "1) Confirm with serial troponins.\n"
            "2) Activate cath lab.\n"
        )
        chunks = segment_text(text)
        self.assertIn("patient has elevated troponin.", [c.lower() for c in chunks])
        self.assertGreaterEqual(len(chunks), 5)

    def test_segment_response_preserves_order(self):
        text = "Step one. Step two. Step three."
        steps = segment_response(text, trajectory_id="A")
        self.assertEqual([s.step_index for s in steps], list(range(len(steps))))
        self.assertTrue(all(s.trajectory_id == "A" for s in steps))
        self.assertEqual(len(steps), 3)

    def test_segment_strips_thinking_tags_and_answer(self):
        text = (
            "<thinking>\n"
            "First the patient presented with chest pain.\n"
            "Then troponin came back elevated.\n"
            "</thinking>\n"
            "<answer>True</answer>\n"
        )
        steps = segment_response(text, trajectory_id="T")
        raw_lower = [s.raw_text.lower() for s in steps]
        self.assertTrue(any("chest pain" in r for r in raw_lower))
        self.assertTrue(any("troponin came back elevated" in r for r in raw_lower))
        self.assertFalse(any("answer" in r and len(r) < 20 for r in raw_lower))

    def test_segment_empty_returns_empty(self):
        self.assertEqual(segment_response("", trajectory_id="x"), [])
        self.assertEqual(segment_response("   \n\n", trajectory_id="x"), [])


if __name__ == "__main__":
    unittest.main()
