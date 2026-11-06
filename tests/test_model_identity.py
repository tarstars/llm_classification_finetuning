import unittest

from arena.kaggle_control import exact_model_version


class ExactModelIdentityTests(unittest.TestCase):
    source = 'qwen-lm/qwen-3/transformers/8b/1'

    def row(self, **changes):
        value = dict(ownerSlug='qwen-lm', modelSlug='qwen-3',
                     framework='MODEL_FRAMEWORK_TRANSFORMERS', variationSlug='8b',
                     versionNumber=1, modelInstanceId=301517, id=363139,
                     url='/models/qwen-lm/qwen-3/Transformers/8b/1')
        value.update(changes)
        return value

    def test_exact_instance_among_same_version_numbers(self):
        desired = self.row()
        others = [self.row(variationSlug='235b-a22b'),
                  self.row(framework='MODEL_FRAMEWORK_GGUF'),
                  self.row(ownerSlug='another-owner'), self.row(modelSlug='another-model'),
                  self.row(versionNumber=2)]
        self.assertIs(exact_model_version(self.source, others + [desired]), desired)
        with self.assertRaisesRegex(ValueError, '0 matches'):
            exact_model_version(self.source, others)

    def test_missing_or_ambiguous_identity_fails(self):
        for field in ['ownerSlug', 'modelSlug', 'framework', 'variationSlug', 'versionNumber']:
            incomplete = self.row()
            del incomplete[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                exact_model_version(self.source, [incomplete])
        with self.assertRaisesRegex(ValueError, '2 matches'):
            exact_model_version(self.source, [self.row(), self.row()])

    def test_url_disagreement_and_unpinned_source_fail(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            exact_model_version(self.source, [self.row(url='/models/qwen-lm/qwen-3/Gguf/8b/1')])
        with self.assertRaises(ValueError):
            exact_model_version(self.source.rsplit('/', 1)[0], [self.row()])


if __name__ == '__main__':
    unittest.main()
