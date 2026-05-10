from torch.utils.data import Sampler
import warnings
import random
import math


warnings.filterwarnings("ignore", message="`data_source` argument is not used", category=UserWarning)


class BucketSampler(Sampler[int]):
    def __init__(self, sentences, batch_size: int) -> None:
        super().__init__(sentences)
        self.sentences = sentences
        self.batch_size = batch_size
        self.length = None

    def __len__(self) -> int:
        if self.length is None:
            self.length = sum(1 for _ in self)
        return self.length

    def __iter__(self):
        indices = list(range(len(self.sentences)))
        indices.sort(key=lambda ind: (len(self.sentences[ind]), random.random()))
        nbuckets = math.ceil(len(indices) / self.batch_size)
        bucket_list = list(range(nbuckets))
        random.shuffle(bucket_list)
        for bucket in bucket_list:
            offset = bucket * self.batch_size
            for index in indices[offset: offset + self.batch_size]:
                yield index
