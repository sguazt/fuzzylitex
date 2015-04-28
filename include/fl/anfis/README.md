## Comparison with MATLAB's anfis

The *MATLAB*'s `anfis` function (or simply *MATLAB*, hereafter) is used to build and train an ANFIS model.
It differs by the ANFIS engine provided in *fuzzylite* in the following:
- In *MATLAB*, the returned trained ANFIS model is the one trained in the penultimate epoch (i.e., the epoch preceding the one when the training procedure stops).
In *fuzzylite*, the returned trained ANFIS model is the one trained in the last epoch run by the training algorithm.
- In *MATLAB*, it is possible to train a model for training data (returned as `fismat1` parameter) and another model for checking data (returned as `fismat2` parameter).
This is not possible in *fuzzylite*.
- In *MATLAB*, it is possible to choose among two different training algorithms, namely *backpropagation* and *hybrid*.
Currently, in *fuzzylite* only the *hybrid* algorithm has been implemented.
