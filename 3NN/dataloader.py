import gzip
import numpy as np
from pathlib import Path
import math

class Dataloader():
    def __init__(self, path, is_train=True, shuffle=True, batch_size=8):
        path = Path(path)
        imagePath = Path(path/'train-images-idx3-ubyte.gz') if is_train else Path(path/'t10k-images-idx3-ubyte.gz')
        labelPath = Path(path/'train-labels-idx1-ubyte.gz') if is_train else Path(path/'t10k-labels-idx1-ubyte.gz')

        self.batch_size = batch_size
        self.images = self.loadImages(imagePath)
        self.labels = self.loadLabels(labelPath)
        self.index = 0
        self.idx = np.arange(0, self.images.shape[0])
        if shuffle: np.random.shuffle(self.idx) # shuffle images

    def __len__(self):
        n_images, _, _, _ = self.images.shape
        n_images = math.ceil(n_images / self.batch_size)
        return n_images

    def __iter__(self):
        return datasetIterator(self)

    def __getitem__(self, index):
        image = self.images[self.idx[index * self.batch_size:(index + 1) * self.batch_size]]
        label = self.labels[self.idx[index * self.batch_size:(index + 1) * self.batch_size]]
        image = image/255.0
        return image, label

    def loadImages(self, path):
        with gzip.open(path) as f:
            images = np.frombuffer(f.read(), 'B', offset=16)
            images = images.reshape(-1, 1, 28, 28).astype(np.float32)
            return images

    def loadLabels(self, path):
        with gzip.open(path) as f:
            labels = np.frombuffer(f.read(), 'B', offset=8)
            rows = len(labels)
            cols = labels.max() + 1
            one_hot = np.zeros((rows, cols)).astype(np.uint8)
            one_hot[np.arange(rows), labels] = 1
            one_hot = one_hot.astype(np.float64)
            return one_hot

# for enumerate magic python function returns Iterator
class datasetIterator():
    def __init__(self, dataloader):
        self.index = 0
        self.dataloader = dataloader

    def __next__(self):
        if self.index < len(self.dataloader):
            item = self.dataloader[self.index]
            self.index += 1
            return item
        # end of iteration
        raise StopIteration
