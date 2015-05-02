# fuzzylitex

Extensions to the [fuzzylite](http://www.fuzzylite.com) project that will be hopefully integrated into the official *fuzzylite* project.


## ANN Extension

This extension provides basic functionalities for *Artificial Neural Networks* (ANN).
This extension is under the `fl::ann` namespace.

### Features

- Multi-layer perceptron
- Learning algorithms:
    - Gradient descent backpropagation learning algorithm with momentum (e.g., see [2]), both with offline and online modality.
- Weight randomizers and initializers:
    - Constant value weight weight randomizer
    - Hard range weight randomizer
    - Nguyen-Widrow's weight randomizer
    - Gaussian weight randomizer

### Limitations


## ANFIS Extension

This extension provides both the *Adaptive Neuro Fuzzy Inference System* (ANFIS) and the related *Coactive ANFIS (CANFIS)* model to support multiple outputs.
For a complete discussion on such models see, for instance, [1].
This extension is under the `fl::anfis` namespace.

### Features

- Multiple outputs
- Learning algorithms:
    - Gradient descent backpropagation learning algorithm with adaptive learning rate based on the step-size (see [1]), both with offline and online modality.
    - Combined gradient descent and least-squares estimation hybrid learning algorithm with adaptive learning rate based on the step-size (see [1]), both with offline and online modality.
- Takagi-Sugeno-Kang and Tsukamoto fuzzy inference system

### Limitations

- The combined gradient descent and least-squares estimation hybrid learning algorithm currently support the Takagi-Sugeno-Kang fuzzy inference system.


## References

1. J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
2. R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
