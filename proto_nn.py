import numpy as np
from util import random_split, batch_data


# Accepts data with the shape (num_of_data_samples, num_of_features)
class ProtoNN:
    def __init__(self, training_data, training_labels, distance, nuclii_size, num_labels, enable_print=False):
        self.training_data = training_data
        self.training_data_size = len(training_data)
        self.training_labels = training_labels
        self.nuclii_size = nuclii_size
        self.sample_size = self.training_data_size - nuclii_size
        self.num_labels = num_labels
        p1, p2 = random_split(nuclii_size, self.training_data_size)
        self.nuclii = training_data[p1]
        self.sample = training_data[p2]
        self.sample_labels = training_labels[p2]
        self.distance = distance
        self.total_voronoi_cells_labels = np.zeros((self.nuclii_size, self.num_labels))
        self.enable_print = enable_print

    def debug(self):
        print('\n'.join([
            f'training_data shape: {self.training_data.shape}',
            f'training_labels shape: {self.training_labels.shape}',
            f'nuclii_size: {self.nuclii_size}',
            f'sample_size: {self.sample_size}',
            f'num_labels: {self.num_labels}',
            f'nuclii shape: {self.nuclii.shape}',
            f'sample shape: {self.sample.shape}',
            f'sample_labels shape: {self.sample_labels.shape}'
        ]))

    def clear_voronoi_cells(self):
        self.total_voronoi_cells_labels = np.zeros((self.nuclii_size, self.num_labels))

    def distribute_voronoi(self, batch_size):
        self.clear_voronoi_cells()
        batches_idx = batch_data(self.sample_size, batch_size)
        for i, batch_idx in enumerate(batches_idx):
            # Distribute batch to voronoi cells
            if ((i % 5 == 0) or (i == (len(batches_idx) - 1))) and self.enable_print:
                print(f'------- {i+1}/{len(batches_idx)} training batches processed -------')
            batch = self.sample[batch_idx]
            batch_labels = self.sample_labels[batch_idx]
            min_distances = self.voronoi_distances(batch).reshape(len(batch), 1)
            voronoi_distribution = (np.arange(self.nuclii_size) == min_distances)
            # sample [0, 0, 1, 0,..., 0]  (1 on its cell)

            # Broadcasting the labels over the second axis (using .T)
            # Adding and subtracting 1 to identify 0 as a label and 0 as False.
            modified_sample_labels = np.broadcast_to(batch_labels + 1, (self.nuclii_size, len(batch))).T
            voronoi_label_distribution = (modified_sample_labels * voronoi_distribution) - 1

            # Count labels of each cell.
            voronoi_cells_labels = np.zeros((self.nuclii_size, self.num_labels))
            for j in range(self.nuclii_size):
                voronoi_cells_labels[j] += self._count_labels(voronoi_label_distribution.T[j])
            self.total_voronoi_cells_labels += voronoi_cells_labels

        self.total_voronoi_cells_labels = self.total_voronoi_cells_labels.argmax(axis=1)
        if self.enable_print:
            print('Finished batches distribution')

    def test(self, test_data, test_labels, batch_size):
        if self.enable_print:
            print('Starting tests')
        batches_idx = batch_data(len(test_data), batch_size)
        error = 0
        current_size = 0
        for i, batch_idx in enumerate(batches_idx):
            batch = test_data[batch_idx]
            batch_labels = test_labels[batch_idx]
            current_size += len(batch_idx)
            min_test_distances = self.voronoi_distances(batch)
            pred_test_labels = self.total_voronoi_cells_labels[min_test_distances]
            error += ((pred_test_labels - batch_labels) != 0).sum()
            if ((i % 5 == 0) or (i == (len(batches_idx) - 1))) and self.enable_print:
                print(f'------- {i+1}/{len(batches_idx)} test batches Processed -------')
                print(f'Current accuracy: {(current_size - error) / current_size}')
        accuracy = (current_size - error) / current_size
        return accuracy

    def voronoi_distances(self, batch):
        batch = batch[:, None, :]
        test_distances = self.distance(batch, self.nuclii)
        return test_distances.argmin(axis=1)

    @staticmethod
    def _count_labels(arr):
        labels, count = np.unique(arr, return_counts=True)
        cell_label_count = np.zeros(10)
        if len(labels) > 1:
            cell_label_count[labels[1:]] += count[1:]
        return cell_label_count

