package main

import (
	"./neural"
	"fmt"
)

func main() {
	nn := neural.CreateNeuralNetwork(2, 2, neural.Step)
	nn.SetConnectionWeight(0, 0, 5.0)
	inputs := []float64{3.0, 2.0}
	fmt.Println(nn.Activate(inputs))
}
