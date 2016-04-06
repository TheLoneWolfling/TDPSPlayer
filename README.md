This is my final project for an AI class - namely, an american-point-system poker squares player using temporal difference learning to train a neural network board state estimator, that in turn feeds a naive (albeit multithreaded) Monte Carlo tree search.

Contained in this repository also are several files used during the development of this project and training of the state estimator used

Contents:

* JHarrisBPANNE.java - this file is the function estimator, with the learning portion removed for speed. Required to run JHarrisTDPlayerMulti.
* JHarrisTDEstimator.dat - this file contains a java serialized JHarrisBPANNE instance, as this was the most convenient way to pass around the weights required. Required to run JHarrisTDPlayerMulti.
* JHarrisTDPlayerMulti.java - this file is the main player, please use it. Note that it requires the two below files.
* DevReport.md - this file contains the development report.

The below files were used during development and are included here for curiosity's sake. Note that they may not compile without minor changes, as files have been renamed since they have been used.

* BPANNE.java - this file is the full version of the function estimator.
* RandTesting.java - this file contains the infrastructure to do initial training of a TDPlayer.
* FinalTweaker.java - this file speculatively adjusts learning rate, potentially backtracking, to make any final tweaks to a function estimator. Generally improves the score by ~5 points.
* TDPlayer.java - this file contains the main temporal difference learning algorithm
