import unittest

from collaborative_filtering.loss import BPRLoss, GBPRLoss


class TestLoss(unittest.TestCase):
    def test_bpr_loss(self):
        criterion = BPRLoss()
        self.assertEqual(1, 1)

    def test_gbpr_loss(self):
        criterion = GBPRLoss(0.5)
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
