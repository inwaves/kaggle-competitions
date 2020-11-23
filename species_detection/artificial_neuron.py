import math

def sigmoid(x):
    y = 1.0 / (1.0 + math.exp(-x))
    return y

def activate(inputs: list, weights: list) -> str:
    # perform net input
    h = 0
    for x, w in zip(inputs, weights):
        h += x*w

    # perform activation
    return sigmoid(h)

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print(output)