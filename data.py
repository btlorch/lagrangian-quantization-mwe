import torchvision
import numpy as np


def load_stl10_dataset(data_dir, transform=None):
    trainset = torchvision.datasets.STL10(root=data_dir, split="train", download=True, transform=transform)
    testset = torchvision.datasets.STL10(root=data_dir, split="test", download=True, transform=transform)
    assert trainset.classes == testset.classes, "Class order is different"

    selected_class_names = ["airplane", "cat"]
    selected_classes = [trainset.classes.index(class_name) for class_name in selected_class_names]

    # Create mapping from previous class target to new class target
    map_old_to_new_class_idx = {source_class_idx: target_class_idx for target_class_idx, source_class_idx in enumerate(selected_classes)}

    # Update training set
    trainset_mask = np.isin(np.array(trainset.labels), selected_classes)
    trainset.classes = selected_class_names
    trainset.data = trainset.data[trainset_mask]
    trainset.labels = [map_old_to_new_class_idx[source_class_idx] for source_class_idx in np.array(trainset.labels)[trainset_mask]]

    # Update test set
    testset_mask = np.isin(np.array(testset.labels), selected_classes)
    testset.classes = selected_class_names
    testset.data = testset.data[testset_mask]
    testset.labels = [map_old_to_new_class_idx[source_class_idx] for source_class_idx in np.array(testset.labels)[testset_mask]]

    return trainset, testset


def load_cifar10_datasets(data_dir, transform=None):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    selected_class_names = ["airplane", "frog"]
    selected_targets = [trainset.class_to_idx[class_name] for class_name in selected_class_names]

    # Create mapping from previous class target to new class target
    map_old_to_new_class_idx = {source_class_idx: target_class_idx for target_class_idx, source_class_idx in enumerate(selected_targets)}
    class_to_idx = {i: class_name for i, class_name in enumerate(selected_class_names)}

    # Update training set
    trainset_mask = np.isin(np.array(trainset.targets), selected_targets)
    trainset.classes = selected_class_names
    trainset.data = trainset.data[trainset_mask]
    trainset.targets = [map_old_to_new_class_idx[source_class_idx] for source_class_idx in np.array(trainset.targets)[trainset_mask]]
    trainset.class_to_idx = class_to_idx

    # Update test set
    testset_mask = np.isin(np.array(testset.targets), selected_targets)
    testset.classes = selected_class_names
    testset.data = testset.data[testset_mask]
    testset.targets = [map_old_to_new_class_idx[source_class_idx] for source_class_idx in np.array(testset.targets)[testset_mask]]
    testset.class_to_idx = class_to_idx

    return trainset, testset
