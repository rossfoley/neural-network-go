package main

import (
	"./neural"
	"fmt"
)

func main() {
	neuralNet := neural.CreateNeuralNetwork([]int{2, 3, 1}, neural.Step)
	neuralNet.SetConnectionWeight(1, 0, 0, 1.0)
	neuralNet.SetConnectionWeight(1, 0, 1, 0.5)
	neuralNet.SetConnectionWeight(1, 1, 1, 0.5)
	neuralNet.SetConnectionWeight(1, 1, 2, 1.0)
	neuralNet.SetConnectionWeight(2, 0, 0, 1.0)
	neuralNet.SetConnectionWeight(2, 1, 0, -2.0)
	neuralNet.SetConnectionWeight(2, 2, 0, 1.0)

	fmt.Println(neuralNet.CreateGraph())
}
