package main

import (
	"./neural"
	"fmt"
)

func main() {
	nnStep := neural.CreateNeuralNetwork(2, 2, neural.Step)
	nnStep.SetConnectionWeight(0, 0, 5.0)
	fmt.Printf("Step function: ")
	fmt.Println(nnStep.Activate([]float64{3.0, 2.0}))

	nnSigmoid := neural.CreateNeuralNetwork(2, 2, neural.Sigmoid)
	nnSigmoid.SetConnectionWeight(0, 0, 5.0)
	fmt.Printf("Sigmoid function: ")
	fmt.Println(nnSigmoid.Activate([]float64{3.0, 2.0}))
}
