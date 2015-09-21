package main

import (
	"./neural"
	"io/ioutil"
)

func main() {
	neuralNet := neural.CreateNeuralNetwork([]int{2, 3, 1}, neural.Step)
	neuralNet.SetDefaultBias(-1.0)
	neuralNet.SetNeuronBias(1, 1, -2.0)
	neuralNet.SetNeuronWeights(1, 0, []float64{1.0, 0.0})
	neuralNet.SetNeuronWeights(1, 1, []float64{1.0, 1.0})
	neuralNet.SetNeuronWeights(1, 2, []float64{0.0, 1.0})
	neuralNet.SetNeuronWeights(2, 0, []float64{1.0, -2.0, 1.0})

	contents := []byte(neuralNet.CreateGraph())
	err := ioutil.WriteFile("neuralnetwork.dot", contents, 0644)
	if err != nil {
		panic(err)
	}
}
