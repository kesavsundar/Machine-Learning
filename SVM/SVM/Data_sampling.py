__author__ = 'kesavsundar'
__author__ = 'kesavsundar'
import mnist as read

class DataSampling():
    def __init__(self):
        self.train_img = None
        self.train_label = None
        self.test_img = None
        self.test_label = None
        self.len_data = 0
        self.sample_len = 0
        self.sample_train_img = None
        self.sample_train_label = None
        return

    def load_data(self):
        reader = read.MNIST(path='dataset/')
        self.train_img, self.train_label = reader.load_training()
        self.test_img, self.test_label = reader.load_testing()
        return

    def do_sampling(self):
        self.sample_len = self.len_data * .03
        each_sample_len = int(self.sample_len / 3)
        sample_data_dict = dict()
        for i in range(0, self.len_data):
            try:
                if len(sample_data_dict[self.train_label[i, :].self.train_label[i, :]]) <each_sample_len:
                    sample_data_dict[self.train_label[i, :]].append(self.train_img[i, :])
            except KeyError:
                sample_data_dict[self.train_label[i, :]] = [self.train_img[i, :]]

        for k, v in zip(sample_data_dict.keys(), sample_data_dict.values()):
            for row in v:

