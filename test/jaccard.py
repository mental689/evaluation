import unittest
from align.jaccard import *
from align.roc import *
import numpy as np
from PIL import Image


class JaccardIndexTest(unittest.TestCase):
    def setUp(self):
        self.y_true = np.ones((10, 3, 100, 100))
        self.y_pred = np.zeros((10, 3, 100, 100))
        self.y_pred[1] = np.ones((3,100,100))

    def test_jaccard(self):
        scores = jaccard_index(self.y_true, self.y_pred)
        assert scores.mean() == 0.1
        precision_recalls = precision_recall(self.y_true, self.y_pred, iou=0.05)
        print ("AP@0.05:{}".format(compute_ap(precision_recalls)))
        print("mAP@0.05:0.95:0.05={}".format(compute_map(self.y_true, self.y_pred)))
    
    def test_jaccard_2(self):
        y_true = np.array(Image.open('data/label.png').convert('L'), dtype=np.float32)[np.newaxis,...] / 255.
        y_pred = np.array(Image.open('data/generation.png').convert('L'), dtype=np.float32)[np.newaxis,...] / 255.
        print('Jaccard Index: {}'.format(jaccard_index(y_true, y_pred)))
        print("mAP@0.05:0.95:0.05={}".format(compute_map(y_true, y_pred)))
if __name__ == '__main__':
    unittest.main()

