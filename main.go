package main

import (
	"./neural"
	"fmt"
)

func main() {
	nn := neural.CreateNeuralNetwork(2, 2, func(sum float64) float64 {
		if sum > 0 {
			return 1.0
		} else {
			return 0.0
		}
	})
	inputs := []float64{1.0, 2.0}
	fmt.Println(nn.Activate(inputs))
}
