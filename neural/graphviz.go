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
		for j := 0; j < layer.size; j++ {
			name := nodeName(layerIndex, j)
			graph.AddNode(name, name, nil)
		}

		for b, weights := range layer.weights {
			for a, weight := range weights {
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
