import torch.nn as nn

def initialize_weights(model: nn.Module, activation: str, init_type: str = "auto") -> str:
    activation = activation.lower()
    init_type = init_type.lower()

    if init_type == "auto":
        if activation in ("relu", "leaky_relu"):
            resolved_init = "he"
        elif activation in ("sigmoid", "tanh"):
            resolved_init = "xavier"
        else:
            resolved_init = "xavier_uniform"
    else:
        resolved_init = init_type

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):

            if resolved_init == "he":
                nn.init.kaiming_normal_(m.weight)

            elif resolved_init == "xavier":
                nn.init.xavier_normal_(m.weight)

            elif resolved_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)

            elif resolved_init == "uniform":
                nn.init.uniform_(m.weight, -0.05, 0.05)

            else:
                raise ValueError(f"Unknown init type: {resolved_init}")

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return resolved_init
