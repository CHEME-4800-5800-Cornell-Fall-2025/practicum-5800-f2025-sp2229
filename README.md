[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LbdSOFva)
# Practicum Problem 4/5800 Fall 2025
In this practicum problem, you will write code to construct, train and test a Classical Hopfield Network (CHN) for the purpose of storing and recalling (uncorrelated) binary images. See `L15a` and `L15c` lecture notes for more details on the classical and modern Hopfield networks.

## Instructions
The student notebook `CHEME-5800-Student-Practicum-ClassicalHopfieldNetworks-F2025.ipynb` has been provided for you. However, all of the underlying source code that makes the notebook work, namely, the types and methods, has been removed. Your task is to re-implement the missing code in the `src/` folder.

Specifically, you will need to implement code in the following files:

> __Code Files to Implement:__
> 
> * `src/Types.jl` - Define any custom types you need for the CHN. We'll use the standard that we have used throughout the class: all types are mutable structs with default constructors. You'll need to define a mutable struct for the Classical Hopfield Network and its associated data.
> * `src/Factory.jl` - Implement code to construct and initialize a Classical Hopfield Network (CHN) type in this file. This includes methods to create the weight matrix using Hebb's rule.
> * `src/Compute.jl` - Implement code to perform recall of stored patterns in this file, and all other utility type methods. This includes the main `recover(...)` method which updates the network state asynchronously and computes performance metrics (e.g., Hamming distance) for convergence analysis.
> 
> If you need to add any helper methods or types, you can create additional files in the `src/` folder as needed.

Finally, make sure to test your code by running the provided notebook. The notebook includes tests in the notebook and at the end that will verify the correctness of your implementation. You are responsible for ensuring that all tests pass and fixing any bugs that arise (including in the notebook itself, if necessary).

## Submission and Grading
Once you have completed the code implementation and the test suite in the notebook runs without errors, submit your entire project to GitHub Classroom as per our typical course submission process. The standard grading rubric will be used to score your practicum submission. 