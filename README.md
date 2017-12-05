# Fuzzylitex

Fuzzylitex is a C++ library for Fuzzy Inference System Modeling and Identification.
It is an extensions to the [fuzzylite](http://www.fuzzylite.com) project that will be hopefully integrated into the official *fuzzylite* project.

## How to Cite

Please, cite this project as the following paper (BibTeX format):

	@MISC{Fuzzylitex,
		author = {Marco Guazzone},
		title = {{Fuzzylitex}: a {C++} library for Fuzzy Inference System Modeling and Identification},
		year = {2015},
		doi = {10.5281/zenodo.592585}
		url = {https://doi.org/10.5281/zenodo.592585},
	}

In addition, you can cite the following papers where Fuzzylitex has been used to carry out the experimental evaluation (BibTeX format):

    @JOURNAL{Prometheus,
        author = {Cosimo Anglano and Massimo Canonico and Marco Guazzone},
        title = {{Prometheus}: a flexible toolkit for the experimentation with virtualized infrastructures},
        journal = {Concurrency and Computation: Practice and Experience},
        year = {2017},
        doi = {10.1002/cpe.4400},
        url = {http://dx.doi.org/10.1002/cpe.4400},
        keywords = {toolkit, resource management, experimental evaluation, physical testbed, virtualization},
        note = {Accepted for publication},
    }


## ANN Extension

This extension provides basic functionalities for *Artificial Neural Networks* (ANN).
This extension is under the `fl::ann` namespace.

### Features

- Multilayer perceptron
- Learning algorithms:
    - Gradient descent with momentum backpropagation learning algorithm (e.g., see [Hagan1996,Rojas1996]), with both offline and online learning mode.
- Weight randomizers and initializers:
    - Constant value weight weight randomizer
    - Hard range weight randomizer
    - Nguyen-Widrow's weight randomizer
    - Gaussian weight randomizer

### Limitations

- Only the multilayer perceptron architecture is available
- Only few variants of the backpropagation algorithm are available

### Requirements

- [Boost C++ library](http://www.boost.org/) and in particular:
   - Boost.CurrentFunction
   - Boost.MPL
   - Boost.Noncopyable
   - Boost.Random (unless a C++11 compiler is used)
   - Boost.Utility (unless a C++11 compiler is used)
   - Boost.TypeErasure
   - Boost.TypeTraits (unless a C++11 compiler is used)

## ANFIS Extension

This extension provides both the *Adaptive Neuro Fuzzy Inference System* (ANFIS) and the related *Coactive ANFIS (CANFIS)* model to support multiple outputs.
For a complete discussion on such models see, for instance, [Jang1993,Jang1997].
This extension is under the `fl::anfis` namespace.

### Features

- Multiple outputs
- Learning algorithms:
    - Gradient descent backpropagation learning algorithm with adaptive learning rate based on the step-size (see [Jang1993,Jang1997]), with both offline and online learning mode.
    - Gradient descent with momentum backpropagation learning algorithm (see [Hagan1996]), with both offline and online learning mode.
    - Combined gradient descent and least-squares estimation hybrid learning algorithm with adaptive learning rate based on the step-size (see [Jang1993,Jang1997]), with both offline and online learning mode.
- Takagi-Sugeno-Kang and Tsukamoto fuzzy inference system

### Limitations

- User defined terms are not supported
- Terms with height other than one are not supported
- The only hedge currently supported in the rule antecedent is the *NOT* hedge.
- The combined gradient descent and least-squares estimation hybrid learning algorithm currently support the Takagi-Sugeno-Kang fuzzy inference system only.


### Requirements

Unless a C++11 compiler is used, the following libraries are needed:

- [Boost C++ library](http://www.boost.org/) and in particular:
   - Boost.CurrentFunction
   - Boost.Utility
   - Boost.TypeTraits


## References

1. *[Hagan1996]* M.T. Hagan et al., "Neural Network Design," Boston, MA: PWS Publishing, 1996.
2. *[Jang1993]* J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
3. *[Jang1997]* J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
4. *[Rojas1996]* R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
