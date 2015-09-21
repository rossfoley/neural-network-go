package main

import (
	"./neural"
	"io/ioutil"
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

	contents := []byte(neuralNet.CreateGraph())
	err := ioutil.WriteFile("neuralnetwork.dot", contents, 0644)
	if err != nil {
		panic(err)
	}
}
