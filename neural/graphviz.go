package neural

import (
	"fmt"
	graphviz "github.com/awalterschulze/gographviz"
)

func (nn NeuralNetwork) CreateGraph() string {
	graph := graphviz.NewGraph()
	graph.SetDir(true)
	graph.SetName("NeuralNetwork")

	for layerIndex, layer := range nn.layers {
		for i, neuron := range layer.neurons {
			name := nodeName(layerIndex, i)
			attrs := make(map[string]string)
			if neuron.bias != 0.0 {
				attrs["label"] = label(name, neuron.bias)
			}
			graph.AddNode(name, name, attrs)
		}

		for b, neuron := range layer.neurons {
			for a, weight := range neuron.weights {
				if weight != 0.0 {
					attrs := make(map[string]string)
					attrs["label"] = fmt.Sprintf("%v", weight)
					aLabel := nodeName(layerIndex-1, a)
					bLabel := nodeName(layerIndex, b)
					graph.AddEdge(aLabel, bLabel, true, attrs)
				}
			}
		}
	}
	return graph.String()
}

func nodeName(layer, node int) string {
	return fmt.Sprintf("L%vN%v", layer, node)
}

func label(name string, bias float64) string {
	return fmt.Sprintf("\"%v, Threshold: %v\"", name, -1.0*bias)
}
