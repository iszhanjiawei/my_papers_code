import ast
import unittest
from pathlib import Path

from grid_wer import normalize
from native_frontend import tree_for, BASELINE, REPA


class NativeFrontendTests(unittest.TestCase):
    def test_only_requested_model_trees(self):
        self.assertEqual(tree_for('baseline'), BASELINE)
        self.assertEqual(tree_for('repa'), REPA)
        for other in ('styledubber','produbber','unknown'):
            with self.assertRaises(ValueError):
                tree_for(other)

    def test_no_comparison_frontend_import(self):
        source = (Path(__file__).parent/'native_frontend.py').read_text()
        names = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                names.extend(alias.name for alias in node.names)
        self.assertFalse({'STYLE','PRO','TOOLS','WEIGHTS'} & set(names))
        self.assertNotIn('stage45_video_embed',source)
        self.assertIn('crop.process_video(',source)

    def test_grid_normalization_is_conservative(self):
        self.assertEqual(normalize('  Place BLUE in U7 please '),'place blue in u seven please')
        self.assertEqual(normalize('a 12 two to too 1'),'a 12 two to too one')
        self.assertEqual(normalize('now no'), 'now no')

    def test_baseline_uses_real_native_cached_prompt_builder(self):
        source = (Path(__file__).parent/'native_baseline.py').read_text()
        self.assertIn('get_inference_prompt_vt(metainfo',source)
        self.assertIn("DATA/'CelebVDub/mel_tacotron/test'",source)
        self.assertIn('use_truth_duration=False',source)
        self.assertNotIn('extractor(audio * scale)',source)


if __name__ == '__main__':
    unittest.main()
