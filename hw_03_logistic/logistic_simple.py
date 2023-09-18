import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LogisticRegressionSimple(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionSimple, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def get_BiNormal():
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    mu1 = -1.5 * torch.ones(2)
    mu2 = 1.5 * torch.ones(2)

    sigma1 = torch.eye(2) * 0.6
    sigma2 = torch.eye(2) * 1.2

    m1 = torch.distributions.MultivariateNormal(mu1, sigma1)
    m2 = torch.distributions.MultivariateNormal(mu2, sigma2)

    x1 = m1.sample((1000,))
    x2 = m2.sample((1000,))

    y1 = np.zeros(x1.size(0))
    y2 = np.ones(x2.size(0))

    X = torch.cat([x1, x2], dim=0)
    Y = np.concatenate([y1, y2])

    return X, Y


def train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=None):
    epochs = 20000

    # Binary Cross-Entropy
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    iter = 0

    for epoch in tqdm(range(int(epochs) + 1), desc='Training Epochs'):
        labels = y_train
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), labels)  # BCE Loss
        loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)

                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test / total_test

                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct / total

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')

    ax1.scatter(X_test[y_test <= 0.5, 0], X_test[y_test <= 0.5, 1])
    ax1.scatter(X_test[y_test > 0.5, 0], X_test[y_test > 0.5, 1])
    ax1.set_title('True data')

    ax2.scatter(X_test[predicted_test <= 0.5, 0], X_test[predicted_test <= 0.5, 1])
    ax2.scatter(X_test[predicted_test > 0.5, 0], X_test[predicted_test > 0.5, 1])

    error_1 = np.logical_and(predicted_test <= 0.5, y_test > 0.5)
    error_2 = np.logical_and(predicted_test > 0.5, y_test <= 0.5)
    ax2.scatter(X_test[error_1, 0], X_test[error_1, 1])
    ax2.scatter(X_test[error_2, 0], X_test[error_2, 1])

    ax2.set_title('Pred data and errors')

    plt.show()


if __name__ == '__main__':
    X, y = get_BiNormal()
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # Two possible outputs
    model = LogisticRegressionSimple(input_dim, output_dim)
    train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=model)
