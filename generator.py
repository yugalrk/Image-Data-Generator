import json
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.index = 0
        self.epoch = 0
        self.label_map = {}

        self.image_paths = [os.path.join(file_path, img) for img in os.listdir(file_path)]
        with open(label_path) as x:
            self.labels = json.load(x)

        for img, label in self.labels.items():
            self.label_map[img] = label

        self.class_name_map = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
            7: 'horse', 8: 'ship', 9: 'truck'}

        if self.shuffle:
            indices = np.random.permutation(len(self.image_paths))
            self.image_paths = np.array(self.image_paths)[indices].tolist()

    def next(self):
        batch_images = []
        batch_labels = []
        start_index = self.index

        for i in range(self.batch_size):
            if self.index >= len(self.image_paths):
                self.index = 0
                self.epoch += 1
                if self.shuffle:
                    indices = np.random.permutation(len(self.image_paths))
                    self.image_paths = np.array(self.image_paths)[indices].tolist()

            image_path = self.image_paths[self.index]
            image = np.load(image_path)

            if self.rotation:
                image = np.rot90(image, k=np.random.randint(1, 4))
            if self.mirroring:
                if np.random.rand() > 0.5:
                    image = np.fliplr(image)

            image = resize(image, self.image_size, mode='reflect')
            batch_images.append(image)
            batch_labels.append(self.label_map[os.path.basename(image_path.split('.npy')[0])])
            self.index += 1

        return np.array(batch_images), np.array(batch_labels)

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        return self.class_name_map[label]

    def show(self):
        images, labels = self.next()
        num_images = images.shape[0]

        fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
        for i in range(num_images):
            title = f'{self.class_name(labels[i])}'
            axes[i].imshow(images[i])
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.show()
