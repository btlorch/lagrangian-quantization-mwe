from data import load_stl10_dataset
import torch
import torchvision.transforms as transforms
import numpy as np
from ddn_with_quantization import DDNQuantizationAttack, NAIVE_ROUND, LAGRANGIAN_QUANTIZATION
from model import Net
from tqdm import tqdm
import pandas as pd
import foolbox
import argparse
import time
import os


def attack(args):
    # Transform images to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load airplane vs. frog datasets
    trainset, testset = load_stl10_dataset(data_dir=args["data_dir"], transform=transform)

    # Set up neural network with two output nodes
    # Net expects input images with intensities in range [0, 1]
    net = Net(num_classes=2)
    net.load_state_dict(torch.load(args["model_filepath"]))

    # Move net onto GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    fmodel = foolbox.models.PyTorchModel(
        model=net,
        bounds=(0., 1.),
    )

    # Lagrangian quantization works with images in range [0, 255].
    # When the Lagrangian quantzation accesses the model (e.g., to evaluate the gradient), the image needs to be transformed from [0, 255] to [0, 1]
    lagrangian_quantization_preprocessing = transforms.Normalize(mean=0, std=255.)

    # Set up DDN attack with quantization
    attack = DDNQuantizationAttack(
        model_torch=net,
        steps=100,
        quantization=args["quantization"],
        confidence=args["confidence"],
        preprocessing=lagrangian_quantization_preprocessing,
        verbose=1,
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    num_test_batches = int(np.ceil(len(testset) / args["batch_size"]))

    if args["max_num_test_batches"]:
        num_test_batches = min(args["max_num_test_batches"], num_test_batches)

    buffer = []
    next_img_idx = 0
    for i, data in tqdm(enumerate(testloader), desc="Iterating test batches", total=num_test_batches):
        if i >= num_test_batches:
            break

        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        current_batch_size = len(labels)

        images = images.to(device)
        labels = labels.to(device)

        # Predictions for original images
        with torch.no_grad():
            y_pred_org = net(images)

        # Run DDN attack
        _, clipped_advs, _ = attack(fmodel, images, labels, epsilons=None)
        assert torch.allclose(clipped_advs * 255, torch.round(clipped_advs * 255), atol=1e-5), "Adversarial images are not quantized"

        # Predictions for adversarial images
        with torch.no_grad():
            y_pred_adv = net(clipped_advs)

        # Calculate attack loss
        # If the attack was successful, this loss should be negative
        loss = y_pred_adv[torch.arange(current_batch_size), labels] - y_pred_adv[torch.arange(current_batch_size), 1 - labels] + attack.confidence

        # L2 distance between original and adverserial images
        l2_distortion = torch.sqrt(torch.mean(torch.flatten(clipped_advs - images, start_dim=1) ** 2, dim=1))

        batch_df = pd.DataFrame({
            "img_idx": np.arange(next_img_idx, next_img_idx + current_batch_size),
            "y_true": labels.detach().cpu().numpy(),
            "y_pred_org_class_0": y_pred_org[:, 0].detach().cpu().numpy(),
            "y_pred_org_class_1": y_pred_org[:, 1].detach().cpu().numpy(),
            "y_pred_adv_class_0": y_pred_adv[:, 0].detach().cpu().numpy(),
            "y_pred_adv_class_1": y_pred_adv[:, 1].detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy(),
            "l2_distortion": l2_distortion.detach().cpu().numpy(),
            "confidence": attack.confidence,
        })

        buffer.append(batch_df)
        next_img_idx += current_batch_size

    attack_df = pd.concat(buffer)

    return attack_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filepath", type=str, help="Path to model filename", default="./stl10_net.pth")
    parser.add_argument("--data_dir", type=str, default="./stl10", help="Path to STL10 directory")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--max_num_test_batches", type=int, default=10, help="Maximum number of test batches")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to store resulting data frame")
    parser.add_argument("--quantization", type=str, default=LAGRANGIAN_QUANTIZATION, choices=[NAIVE_ROUND, LAGRANGIAN_QUANTIZATION], help="Quantization method")
    parser.add_argument("--confidence", type=str, default=0, help="Attack confidence")

    args = vars(parser.parse_args())

    attack_df = attack(args)

    output_filepath = os.path.join(args["output_dir"], f"{time.strftime('%Y_%m_%d')}_stl10_quantization_{args['quantization']}_confidence_{args['confidence']}.csv")
    print(f"Output file: \"{output_filepath}\"")
    attack_df.to_csv(output_filepath, index=False)
