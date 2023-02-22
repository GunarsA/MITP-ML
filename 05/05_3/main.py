import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7)  # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


def normalize(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    return (data - data_min) / (data_max - data_min)


class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        # x_brands,
        # x_fuel,
        # x_transmission,
        # x_seller_type,
        # x_year,
        # x_km_drivers

        self.X = np.array(self.X, dtype=np.float32)
        self.X = normalize(self.X)

        # y_owner
        # y_selling_price

        self.Y = np.array(self.Y, dtype=np.float64)
        self.Y = normalize(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.Y[idx])


class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration

        # Define start and end indexes
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size

        # Sample X and Y
        x, y = self.dataset[idx_start:idx_end]

        # Increment batch idx value
        self.idx_batch += 1
        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W: Variable = Variable(
            value=np.random.uniform(low=-1., size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.zeros(shape=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        # W.shape = (in, out)
        # x.shape = (B, in)
        # output.shape = (B, out)
        self.output = Variable(
            np.squeeze(self.W.value.T @ np.expand_dims(self.x.value, axis=-1), axis=-1) + self.b.value
        )
        return self.output

    def backward(self):
        # How does Linear change as I change b?

        # dLinear/db = 1 * chain rule
        self.b.grad += 1 * self.output.grad

        # dLinear/dW = x * chain_rule
        # x.value.shape = (B, in, 1)
        # output.grad.shape = (B, 1, out)
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)

        # dLinear/dx = W
        # W.shape = (in, out)
        # output.grad.shape = (B, out, 1)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)


class LayerSigmoid:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1.0 / (1.0 + np.exp(-self.x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1.0 - self.output.value) * self.output.grad


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x  # TODO
        self.output = None
        return self.output

    def backward(self):
        self.x.grad += 1  # TODO


class LossMSE:
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = 0  # TODO
        return loss

    def backward(self):
        self.y_prim.grad += 1  # TODO


class LossMAE:
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(self.y.value - self.y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad += -(self.y.value - self.y_prim.value) / \
                            (np.abs(self.y.value - self.y_prim.value) + 1e-8)


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=6, out_features=4),
            LayerSigmoid(),
            LayerLinear(in_features=4, out_features=4),
            LayerSigmoid(),
            LayerLinear(in_features=4, out_features=2)
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


# For later
class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x_indexes = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.random.random((num_embeddings, embedding_dim)))
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x_indexes = x.value.squeeze().astype(np.int)
        self.output = Variable(
            np.array(self.emb_m.value[self.x_indexes, :]))  # same as dot product with one-hot encoded X and Emb_w
        return self.output

    def backward(self):
        self.emb_m.grad[self.x_indexes, :] += self.output.grad


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMAE()

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, y in dataloader:

            y_prim = model.forward(x=Variable(value=x))
            loss = loss_fn.forward(y=Variable(value=y), y_prim=y_prim)

            losses.append(loss)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 100 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
