# RNOG Image Builder - Research Project
## Research Project under Prof. Brian Clark
University of Maryland
Department of Physics

**Project Overview**

This repository documents the research and development of machine learning methods—specifically Convolutional Neural Networks (CNNs)—to reconstruct neutrino events using simulated data from the Radio Neutrino Observatory in Greenland (RNO-G). The primary goal is to interpret voltage pulses detected by phased antenna arrays to accurately classify and reconstruct high-energy particle events.

**Images?**

The term _image_ stems from the first iteration of this research's objective. At the beginning, we wanted to be able to convey the voltage pulses detected from a neutrino event at RNO-G in a compact manner so that it could be fed into a Convolutional Neural Network.

We came up with the idea of representing these voltages as a matrix of pixels (hence "image"), where each pixel is a binned voltage value and the width and height represent different times and channels. Below is a picture adapted from a poster I presented for reference:

<img width="757" height="1070" alt="image" src="https://github.com/user-attachments/assets/8c153b42-3ea0-4208-84b7-ed984b2b9e86" />

As the project developed, these images are now 3 dimensional! Where the new dimension represents a separate station, a factor that we had not previously considered given its complexity.

**Repository Structure**

Since this is primarily a storage repo for my ongoing research, the code is organized by the different stages of the pipeline:

jobs/: This directory contains all the job files needed to run thousands of simulations on a High-Performance Computing (HPC) cluster using HTCondor.

machine_learning/: This directory is where the learning happens. It contains various iterations of code exploring different CNN architectures and hyperparameters aimed at minimizing the loss function.

Simulation/: This directory houses the core simulation logic. This code was adapted from the example provided by my mentor Brian Clark.

Utils/: This directory contains custom, reusable code written to streamline many repetitive actions across the entire pipeline.
