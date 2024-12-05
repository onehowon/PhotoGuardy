from art.estimators.classification import PyTorchClassifier
import torch.optim as optim

def create_classifier(model, criterion, learning_rate=0.001, input_shape=(3, 224, 224), nb_classes=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )
