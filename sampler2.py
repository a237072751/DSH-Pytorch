import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import random

class categoryRandomSampler(Sampler):
    def __init__(self, numBatchCategory, batch_size, targets_path):
        """
        This sampler will sample numBatchCategory categories in each batch.
        """
        with open(targets_path) as f:
            targets = [int(line.split()[-1].strip()) for line in f]

        self.targets = list(targets)
        self.batch_size = batch_size
        self.num_samples = len(targets)
        self.numBatchCategory = numBatchCategory
        self.num_categories = len(list(set(targets)))
        self.category_idxs = {}
        self.categorys = list(set(targets))

        for category in self.categorys:
            self.category_idxs[category] = []

        for i in range(self.num_samples):
            self.category_idxs[targets[i]].append(i)

    def __iter__(self):
        num_batches = self.num_samples // self.batch_size
        selected = []
        for i in range(num_batches):
            batch = []
            random.shuffle(self.categorys)
            categories_selcted = self.categorys[:self.numBatchCategory]
            # categories_selcted = np.random.randint(self.num_categories, size=self.numBatchCategory)

            for j in categories_selcted:
                random.shuffle(self.category_idxs[j])
                batch.extend(self.category_idxs[j][:int(self.batch_size // self.numBatchCategory)])

            random.shuffle(batch)
            selected.extend(batch)

        return iter(torch.LongTensor(selected))

    def __len__(self):
        return self.num_samples
