package main

import (
	"./neural"
	"fmt"
)

func main() {
	nnStep := neural.CreateNeuralNetwork([]int{2, 2}, neural.Step)
	nnStep.SetConnectionWeight(1, 0, 0, 5.0)
	fmt.Printf("Step function: ")
	fmt.Println(nnStep.Activate([]float64{3.0, 2.0}))

	nnSigmoid := neural.CreateNeuralNetwork([]int{2, 2}, neural.Sigmoid)
	nnSigmoid.SetConnectionWeight(1, 0, 0, 5.0)
	fmt.Printf("Sigmoid function: ")
	fmt.Println(nnSigmoid.Activate([]float64{3.0, 2.0}))

	nnTanhSigmoid := neural.CreateNeuralNetwork([]int{2, 2}, neural.TanhSigmoid)
	nnTanhSigmoid.SetConnectionWeight(1, 0, 0, 5.0)
	fmt.Printf("Tanh Sigmoid function: ")
	fmt.Println(nnTanhSigmoid.Activate([]float64{3.0, 2.0}))

	fmt.Println("GraphViz output for step")
	fmt.Println(nnStep.CreateGraph())
}
