# Space

Code I'm working on for learning/entertainment purposes.

## Contents

The .py files in the main directory contain an interplanetary trajectory optimiser.
- KepEq.py contains functions calculating eccentric/hyperbolic anomalies.
- LambSolve.py contains a Lambert's problem solver adapted from https://arxiv.org/abs/1403.2705
- Optimiser.py contains a genetic algorithm optimiser.
- trajViz.py contains tools to visualise the obtained trajectory
- Bodies.py (WIP) contains classes representing:
  - Spacecrafts
	- Celestial bodies (star, planets, moons)
- Example.py gives an instance of a problem to optimise, using the [Kerbal Space Program](https://www.kerbalspaceprogram.com/) solar system

The optimiser is based on the [MGA-1DSM problem](https://pdfs.semanticscholar.org/5ca7/dec2d84dc269921fa19d357b07af7f341f30.pdf). To sum up, it assumes a patched conics approximation where the Sphere Of Influences (SOI) have a radius of 0. Optimisation is done assuming a preset list of flybys. Spacecrafts are allowed one Deep-Space Maneuver (DSM) between each flyby.
The "old" folder contains another (not really working) optimiser.

The attitude control folder contains tools for orientations computation (ie Euler angles to rotation matrix) made during the [Spacecraft Dynamics and Control MOOCs](Spacecraft Dynamics and Control).