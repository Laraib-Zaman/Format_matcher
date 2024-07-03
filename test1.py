# Usage example:
from fm_iter3 import ImageSimilarityDetector
detector = ImageSimilarityDetector()
top_indexes, imgs, ssim, cluster = detector.incept('/content/drive/MyDrive/eob_small/ss1.png', 0.60, '/content/drive/MyDrive/eob_small/eobs.zip')
