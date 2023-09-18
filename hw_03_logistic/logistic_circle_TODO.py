import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from tqdm import tqdm
import torch
from logistic_simple import train


class LogisticRegressionCircle(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionCircle, self).__init__()
        # DO NOT MODIFY THIS
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        ######################################
        ##### Modify the model here ##########
        ######################################
        # TODO: modify forward function
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def get_circle():
    np.random.seed(123)
    X = np.random.normal(0, 1, size=(1000, 2))
    y = np.array((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5, dtype='int')
    return X, y


if __name__ == '__main__':
    ###############################
    ####     DO NOT MODIFY    #####
    ###############################
    X, y = get_circle()
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # Two possible outputs
    model = LogisticRegressionCircle(input_dim, output_dim)
    ###############################
    ####     DO NOT MODIFY    #####
    ###############################

    # TODO: you may modify your learning_rate
    # But do not change train() function
    train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=model)
