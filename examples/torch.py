"""Train a small PyTorch model from a PixelPipes stream."""

try:
    import torch
except ImportError as error:
    raise SystemExit("Install PyTorch to run this example: python -m pip install torch") from error

from pixelpipes.graph import SampleIndex
from pixelpipes.sink import TorchDataLoader
from pixelpipes.tensor import RandomTensor
from pixelpipes.utilities import pipeline


@pipeline()
def samples():
    features = RandomTensor([6], seed=SampleIndex())
    return features, features


def main():
    loader = TorchDataLoader(samples(), batch=8, workers=1)
    model = torch.nn.Linear(6, 6)
    optimizer = torch.optim.Adam(model.parameters())

    for batch, _ in zip(loader, range(2)):
        features, targets = batch
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()

    print("PyTorch training completed")


if __name__ == "__main__":
    main()
