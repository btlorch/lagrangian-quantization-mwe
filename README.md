# DDN attack with Lagrangian quantization

This is a minimal working example that demonstrates the DDN attack against a simple CNN-based binary classifier.

The CNN is trained on airplane and cat images from the STL10 dataset. After 40 training epochs, the CNN achieves a test accuracy of 95%.

## Steps

1. Train the CNN

```bash
python train.py
```

Optional arguments:
* ``--data_dir``: Directory where to store STL10 images. If this directory does not exist, it will be created and the images will be downloaded automatically.
* ``--output_dir``: Directory where to store the trained model.
* ``--num_epochs``: Number of training epochs, default is 40.
* ``--batch_size``: Number of images per batch, default is 64.

2. Attack the CNN

```bash
python attack.py
```

Useful arguments (please look at the script to see all arguments):
* ``--data_dir``: Directory where to find STL10 images. If this directory does not exist, it will be created and the images will be downloaded automatically.
* ``--output_dir``: Directory where to store the results as csv file.
* ``--quantization``: Which method to use for quantization, either "naive_round" or "lagrangian_quantization".
* ``--confidence``: The attack succeeds after enforcing the given margin between the logits corresponding to the wrong class and the true class. Set to 0 by default. 

## Results

Attack 100 test set images. The DDN used 500 steps, and the default hyper-parameters (`init_epislon = 1.0`, `gamma = 0.05`).

| Confidence | Quantization | Attack success | L2 distortion |
| --- | --- | --- | --- |
|  0 | Naive round | 1.0 | 0.00831 |
|  0 | Lagrangian  | 1.0 | 0.00842 |
| --- | --- | --- | --- |
| 10 | Naive round | 1.0 | 0.01469 |
| 10 | Lagrangian  | 1.0 | 0.01481 |
| --- | --- | --- | --- |
| 20 | Naive round | 1.0 | 0.02700 |
| 20 | Lagrangian  | 1.0 | 0.02700 |
