import torch

def choose_optimizer(model, cfg):
    optimizer = cfg["optimizer"]
    lr = cfg["training"]["learning_rate"]

    if optimizer["type"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )
    
    if optimizer["type"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
        )

    if optimizer["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer.get("momentum", 0.9),
        )

    if optimizer["type"] == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=optimizer.get("momentum", 0.9),
        )
    
    if optimizer["type"] == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
        )

    else:
        raise ValueError(f"Unknown optimizer type: {cfg['type']}")
