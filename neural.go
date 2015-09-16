package main

import "fmt"

type NeuralNetwork struct {
	numInputs      int
	outputs        []Node
	activationFunc func(float64) float64
}

type Node struct {
	weights []float64
}

func (n Node) sum(inputs []float64) float64 {
	sum := 0
	for i, v := range n.weights {
		sum += inputs[i] * v
	}
	return sum
}

func (nn NeuralNetwork) activate(inputs []float64) []float64 {
	var result [len(nn.outputs)]float64
	for i, v := range nn.outputs {
		sum := v.sum(inputs)
		result[i] = nn.activationFunc(sum)
	}
	return result
}
