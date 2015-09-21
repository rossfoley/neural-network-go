package neural

import "math"

func Step(sum float64) float64 {
	if sum >= 0.0 {
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
