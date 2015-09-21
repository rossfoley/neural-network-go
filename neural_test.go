package main

import (
	"./neural"
	"testing"
)

type testpair struct {
	inputs []float64
	output float64
}

var xorTests = []testpair{
	{[]float64{1.0, 1.0}, 0.0},
	{[]float64{1.0, 0.0}, 1.0},
	{[]float64{0.0, 1.0}, 1.0},
	{[]float64{0.0, 0.0}, 0.0},
}

func TestXOR(t *testing.T) {
	// Setup the structure and weights of the XOR neural network
	neuralNet := neural.CreateNeuralNetwork([]int{2, 3, 1}, neural.Step)
	neuralNet.SetConnectionWeight(1, 0, 0, 1.0)
	neuralNet.SetConnectionWeight(1, 0, 1, 0.5)
	neuralNet.SetConnectionWeight(1, 1, 1, 0.5)
	neuralNet.SetConnectionWeight(1, 1, 2, 1.0)
	neuralNet.SetConnectionWeight(2, 0, 0, 1.0)
	neuralNet.SetConnectionWeight(2, 1, 0, -2.0)
	neuralNet.SetConnectionWeight(2, 2, 0, 1.0)

	for _, pair := range xorTests {
		output := neuralNet.Activate(pair.inputs)[0]
		if output != pair.output {
			t.Error(
				"For", pair.inputs,
				"expected", pair.output,
				"got", output,
			)
		}
	}
}
