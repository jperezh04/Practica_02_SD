import ray
import torch
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ray.train import ScalingConfig
from ray.train.torch import (
    TorchTrainer,
    prepare_model,
    prepare_data_loader
)

from ray import train

from model import NeuralNet


def train_func(config):

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    dataloader = prepare_data_loader(dataloader)

    model = NeuralNet()
    model = prepare_model(model)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 3

    for epoch in range(epochs):

        total_loss = 0.0

        for images, labels in dataloader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train.report(
            {
                "epoch": epoch,
                "loss": total_loss
            }
        )


if __name__ == "__main__":

    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False
        )
    )

    result = trainer.fit()

    print("\n=== RESULTADOS ===")
    print(result.metrics)

    ray.shutdown()