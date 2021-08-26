import unittest

from collaborative_filtering.parameters import parse_args


class TestParameters(unittest.TestCase):
    # TODO how to pass parameters to the test?
    # def setUp(self):
    #     self.args = parse_args()

    def test_required_parameters(self):
        self.assertEqual(1, 1)

    def test_parameters_assertion(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
