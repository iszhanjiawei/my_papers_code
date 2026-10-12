import unittest
from collections import Counter
from aligndit.script.eval.grid_s1_common import identity, normalize_text
from aligndit.script.eval.grid_s2_common import select_references


def row(speaker, utterance, text):
    return {'speaker': speaker, 'audio_relative_path': f'val/{speaker}/{utterance}.wav',
            'utterance_key': f'grid/{speaker}/{utterance}', 'text': text}


class GridS2EvaluationTests(unittest.TestCase):
    def test_different_text_same_speaker_permutation_is_reproducible(self):
        rows = [row(s, f'clip{i}', f'bin white by {i} two again') for s in ('s1','s2') for i in range(7)]
        first, second = select_references(rows), select_references(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(Counter(identity(r) for r in first.values()), Counter(identity(r) for r in rows))
        for target in rows:
            ref = first[identity(target)]
            self.assertNotEqual(identity(target), identity(ref))
            self.assertNotEqual(normalize_text(target['text']), normalize_text(ref['text']))
            self.assertEqual(target['speaker'], ref['speaker'])

    def test_duplicate_dialogue_is_never_the_reference(self):
        rows = [row('s1', f'clip{i}', text) for i,text in enumerate(['A two','a 2','B three','C four','D five','E six'])]
        for target in rows:
            self.assertNotEqual(normalize_text(target['text']), normalize_text(select_references(rows)[identity(target)]['text']))

    def test_impossible_independent_reference_is_rejected(self):
        for rows in ([row('s1','x','a')], [row('s1','x','a'),row('s1','y','A')]):
            with self.assertRaises(ValueError):
                select_references(rows)


if __name__ == '__main__':
    unittest.main()
