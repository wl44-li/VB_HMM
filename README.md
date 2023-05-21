# VB_HMM
Variational Bayesian Inference for Hidden Markov Models

## Important: Work in Progress
- For latest supported features, please refer to the `development` branch
- Under development: 
  - Batch and Stochastic optimisation comparison
  - Non-parametric extension, i.e. infinite Hidden Markov Models

## Setup and Installation
You need to install [Julia](https://julialang.org/) (1.9) and then [Pluto](https://plutojl.org/) to run the notebook.

Alternatively, static html files are availble for general browsing using any internet browser.

## Usage
On your command prompt or terminal:
```bash
julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.0 (2023-05-07)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> using Pluto

julia> Pluto.run()
[ Info: Loading...
┌ Info: 
└ Opening http://localhost:1234/?secret=$$$$$ in your default browser... ~ have fun!
┌ Info: 
│ Press Ctrl+C in this terminal to stop Pluto
└ 

```
Pluto should launch on your default browser, you can then use it to open the designated notebook file (i.e. `VBEM_HMM_fox.jl`). 


## Credits
This work was inspired by or uses the following sources:

- Beal, Matthew J. (2003). Variational Algorithm for Approximate Bayesian Inference. UCL.

- Fox et al. (2014). Stochastic Variational Inference for Hidden Markov Models. 

