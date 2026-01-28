# Voice Recognition CNN - Configuration Guide
This project uses YAML configuration files to control model architecture, training parameters, and optimization settings.
Configuration File Structure
Create a `config.yaml` file in your project root with the following structure:

```yaml
optimizer:
  type: adam | adamw | sgd | rmsprop | adagrad
  momentum: 0.9  # Only used for SGD and RMSprop

training:
  learning_rate: 0.001

model:
  dropout: 0.3
  batch_norm: after | before | none
  activation: relu | leaky_relu | sigmoid | tanh
  initialize_weights: auto | he | xavier | uniform
```

