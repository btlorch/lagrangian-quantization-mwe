# DDN attack with Lagrangian quantization

This is a minimal working example that demonstrates the DDN attack against a simple CNN-based binary classifier.

The CNN is trained on airplane and cat images from the STL10 dataset. After 40 training epochs, the CNN achieves a test accuracy of 95%.

## Steps

### 1. Train the CNN (You can skip this step and use the pre-trained `stl10_net.pth`)

```bash
python train.py
```

Optional arguments:
* ``--data_dir``: Directory where to store STL10 images. If this directory does not exist, it will be created and the images will be downloaded automatically.
* ``--output_dir``: Directory where to store the trained model.
* ``--num_epochs``: Number of training epochs, default is 40.
* ``--batch_size``: Number of images per batch, default is 64.

### 2. Attack the CNN

Compared to the DDN attack provided by Foolbox, this implementation adds:
- Quantization add the end of each attack iteration
- Confidence (sometimes called safety margin): An image is adversarial once the logits corresponding to the wrong and the right class differ by some margin.
- Option to use another loss more similar to the C&W attack. Not used right now.

```bash
python attack.py
```

Useful arguments (please look at the script to see all arguments):
* ``--data_dir``: Directory where to find STL10 images. If this directory does not exist, it will be created and the images will be downloaded automatically.
* ``--output_dir``: Directory where to store the results as csv file.
* ``--quantization``: Which method to use for quantization, either "naive_round" or "lagrangian_quantization".
* ``--confidence``: The attack succeeds after enforcing the given margin between the logits corresponding to the wrong class and the true class. Set to 0 by default.

## Results

Attack 100 test set images with DDN default hyper-parameters (`init_epsilon = 1.0`, `gamma = 0.05`). The table shows the results after running the attack with 100 and 500 iterations. In all cases the attack succeeded.

| Confidence | Quantization | L2 distortion (100 iterations) | L2 distortion (500 iterations) |
| --- | --- | --- | --- |
|  0 | Naive round | 0.010359 | 0.008312 |
|  0 | Lagrangian  | 0.008477 | 0.008418 |
| --- | --- | --- | --- |
| 10 | Naive round | 0.017106 | 0.014679 |
| 10 | Lagrangian  | 0.014906 | 0.014811 |
| --- | --- | --- | --- |
| 20 | Naive round | 0.029708 | 0.026994 |
| 20 | Lagrangian  | 0.027228 | 0.026998 |

Observation: With a limited number of attack iterations, Lagrangian quantization outperforms naive rounding. Given enough attack iterations, naive rounding achieves as low distortion as Lagrangian quantization (on average even slightly lower than Lagrangian quantization).