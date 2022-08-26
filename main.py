from util import get_data, pingpong, l2_distance
from proto_nn import ProtoNN
from math import floor

TRAIN_IMAGES_SIZE = 1500
TEST_IMAGES_SIZE = 1000
NUCLII_SIZE = floor(TRAIN_IMAGES_SIZE * (5/100))
SAMPLE_SIZE = TRAIN_IMAGES_SIZE - NUCLII_SIZE
BATCH_SIZE = 64


def main():
    train_images, train_labels = get_data(size=TRAIN_IMAGES_SIZE)
    test_images, test_labels = get_data(size=TEST_IMAGES_SIZE, train=False)

    x = ProtoNN(train_images, train_labels, l2_distance, nuclii_size=NUCLII_SIZE, num_labels=10, enable_print=True)
    x.distribute_voronoi(BATCH_SIZE)
    acc = x.test(test_images, test_labels, batch_size=85)
    print(f'Test accuracy:{acc}')


if __name__ == '__main__':
    with pingpong('main'):
        main()

