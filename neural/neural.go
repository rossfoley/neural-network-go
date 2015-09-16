package neural

type NeuralNetwork struct {
	numInputs      int
	outputs        []Node
	activationFunc func(float64) float64
}

type Node struct {
	weights []float64
}

func (n Node) sum(inputs []float64) float64 {
	sum := 0.0
	for i, v := range n.weights {
		sum += inputs[i] * v
	}
	return sum
}

func (nn NeuralNetwork) Activate(inputs []float64) []float64 {
	var result = make([]float64, len(nn.outputs))
	for i, v := range nn.outputs {
		sum := v.sum(inputs)
		result[i] = nn.activationFunc(sum)
	}
	return result
}

func CreateNeuralNetwork(inputs int, outputs int, activationFunc func(float64) float64) NeuralNetwork {
	var outputNodes = make([]Node, outputs)
	for i := 0; i < outputs; i++ {
		var weights = make([]float64, inputs)
		outputNodes[i] = Node{weights}
	}
	return NeuralNetwork{inputs, outputNodes, activationFunc}
}
