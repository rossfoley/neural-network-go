package neural

import "math"

/* Structs */

type NeuralNetwork struct {
	layers         []Layer
	activationFunc func(float64) float64
}

type Layer struct {
	size    int
	weights [][]float64
}

/* NeuralNetwork methods */

func (nn NeuralNetwork) Activate(inputValues []float64) []float64 {
	var inputs = inputValues

	// Loop through the weights, ignoring the first layer (input nodes)
	for i := 1; i < len(nn.layers); i++ {
		layer := nn.layers[i]
		outputs := make([]float64, layer.size)

		for j, weights := range layer.weights {
			sum := 0.0
			for k, weight := range weights {
				sum += inputs[k] * weight
			}
			outputs[j] = nn.activationFunc(sum)
		}

		// The new output values become the input for the next layer
		inputs = outputs
	}

	// The last value for inputs is the output of the output nodes
	return inputs
}

func (nn *NeuralNetwork) SetConnectionWeight(layer int, input int, node int, weight float64) {
	nn.layers[layer].weights[node][input] = weight
}

/* Constructor */

func CreateNeuralNetwork(shape []int, activationFunc func(float64) float64) NeuralNetwork {
	layers := make([]Layer, len(shape))
	outputSize := shape[0]

	for i, size := range shape {
		weights := make([][]float64, size)
		for j := range weights {
			weights[j] = make([]float64, outputSize)
		}
		layers[i] = Layer{size, weights}
		outputSize = size
	}

	return NeuralNetwork{layers, activationFunc}
}

/* Activation functions */

func Step(sum float64) float64 {
	if sum > 0 {
		return 1.0
	} else {
		return 0.0
	}
}

func Sigmoid(sum float64) float64 {
	return 1 / (1 + math.Pow(math.E, -1.0*sum))
}

func TanhSigmoid(sum float64) float64 {
	return math.Tanh(sum)
}

/* Helper functions */

func max(values []int) int {
	maxInt := 0
	for _, v := range values {
		if v > maxInt {
			maxInt = v
		}
	}
	return maxInt
}
