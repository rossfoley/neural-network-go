package neural

import "math"

/* Structs */
type NeuralNetwork struct {
	numInputs      int
	outputs        []Node
	activationFunc func(float64) float64
}

type Node struct {
	weights []float64
}

/* NeuralNetwork methods */
func (nn NeuralNetwork) Activate(inputs []float64) []float64 {
	var result = make([]float64, len(nn.outputs))
	for i, v := range nn.outputs {
		sum := v.sum(inputs)
		result[i] = nn.activationFunc(sum)
	}
	return result
}

func (nn *NeuralNetwork) SetConnectionWeight(input int, output int, weight float64) {
	nn.outputs[output].weights[input] = weight
}

/* Node methods */
func (n Node) sum(inputs []float64) float64 {
	sum := 0.0
	for i, v := range n.weights {
		sum += inputs[i] * v
	}
	return sum
}

/* Helper methods */
func CreateNeuralNetwork(inputs int, outputs int, activationFunc func(float64) float64) NeuralNetwork {
	var outputNodes = make([]Node, outputs)
	for i := 0; i < outputs; i++ {
		var weights = make([]float64, inputs)
		outputNodes[i] = Node{weights}
	}
	return NeuralNetwork{inputs, outputNodes, activationFunc}
}

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
