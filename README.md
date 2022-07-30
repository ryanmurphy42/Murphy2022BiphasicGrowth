# Murphy et al. (2022) Computationally efficient framework for diagnosing, understanding, and predicting biphasic population growth.

Preprint available on bioRxiv: https://doi.org/10.1101/2022.07.27.501797 

This repository holds key Julia code and all experimental data used to generate figures in the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in July 2022 using:

- Julia Version  1.7.2 (see https://julialang.org/downloads/ )
- Julia packages: Plots, LinearAlgebra, NLopt, .Threads, Interpolations, Distributions, Roots, LaTeXStrings

## Guide to using the code
There are four scripts. Each script contains experimental data, estimates the MLE, estimates the change point and other model parameters, and parameter-wise profile predictions. Figures are generated showing: comparisons of the experimental with the mathematical model simulated with the MLE; profile likelihoods for the change point and other model parameters; parameter-wise profile predictions; and the difference between the parameter-wise profile predictions and the mathematical model simulated with the MLE.

The four scripts are:
1. Biphasic_001_Coral.jl
2. Biphasic_002_2DProliferationAssay.jl
3. Biphasic_003_3DSpheroid.jl
4. Biphasic_004_2DProliferationAssayBladder.jl
