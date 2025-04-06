package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type GameState struct {
	board [9]string
	currentPlayer int // 0 for X(player), 1 for O (computer)
}

type NeuralNetwork struct {
	inputSize int
	outputSize int
	hiddenSize int
	weightsIH []float64
	weightsHO []float64
	biasesH []float64
	biasesO []float64

	inputs []float64
	hidden []float64
	rawLogits []float64 // logits before applying softmax
	outputs []float64
}

func newNeuralNetwork(inputSize, outputSize, hiddenSize int) *NeuralNetwork {
	return &NeuralNetwork{
		inputSize: inputSize,
		outputSize: outputSize,
		hiddenSize: hiddenSize,
		weightsIH: make([]float64, inputSize*hiddenSize),
		weightsHO: make([]float64, hiddenSize*outputSize),
		biasesH: make([]float64, hiddenSize),
		biasesO: make([]float64, outputSize),
	}
}

func (nn *NeuralNetwork) setWeights(weightsIH, weightsHO, biasesH, biasesO []float64) {
	nn.weightsIH = weightsIH
	nn.weightsHO = weightsHO
	nn.biasesH = biasesH
	nn.biasesO = biasesO
}

func reLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func reLUDerivative(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func initNN(inputSize, outputSize, hiddenSize int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	nn := newNeuralNetwork(inputSize, outputSize, hiddenSize)
	for i := 0; i < nn.inputSize*nn.hiddenSize; i++ {
		nn.weightsIH[i] = rand.Float64()*2 - 1
	}
	for i := 0; i < nn.hiddenSize*nn.outputSize; i++ {
		nn.weightsHO[i] = rand.Float64()*2 - 1
	}
	for i := 0; i < nn.hiddenSize; i++ {
		nn.biasesH[i] = rand.Float64()*2 - 1
	}
	for i := 0; i < nn.outputSize; i++ {
		nn.biasesO[i] = rand.Float64()*2 - 1
	}
	return nn
}

func softmax(input, output *[]float64) {
	var max float64
	max = (*input)[0]
	for _, v := range *input {
		if v > max {
			max = v
		}
	}
	for i, v := range *input {
		(*output)[i] = math.Exp(v - max)
	}
	var sum float64
	for _, v := range *output {
		sum += v
	}
	if sum > 0 {
		for i, _ := range *output {
			(*output)[i] /= sum
		}
	} else {
		for i, _ := range *output {
			(*output)[i] = 1.0 / float64(len(*output))
		}
	}
}

func (nn *NeuralNetwork) forwardPass(input []float64) {
	// input
	nn.inputs = input
	// hidden
	for i := 0; i < nn.hiddenSize; i++ {
		var sum float64
		for j := 0; j < nn.inputSize; j++ {
			sum += nn.inputs[j] * nn.weightsIH[i*nn.inputSize+j]
		}
		nn.hidden[i] = reLU(sum + nn.biasesH[i])
	}
	// output
	for i := 0; i < nn.outputSize; i++ {
		var sum float64
		for j := 0; j < nn.hiddenSize; j++ {
			sum += nn.hidden[j] * nn.weightsHO[i*nn.hiddenSize+j]
		}
		nn.rawLogits[i] = reLU(sum + nn.biasesO[i])
	}

	softmax(&nn.rawLogits, &nn.outputs)
}

func initGame() *GameState {
	return &GameState{
		board: [9]string{".", ".", ".", ".", ".", ".", ".", ".", "."},
		currentPlayer: 0,
	}	
} 

func displayBoard(board [9]string) {
	fmt.Println(board[0], board[1], board[2])
	fmt.Println(board[3], board[4], board[5])
	fmt.Println(board[6], board[7], board[8])
}

func (gs *GameState) boardToInputs() []float64 {
	inputs := make([]float64, 9)
	for i := 0; i < 9; i++ {
		if gs.board[i] == "X" {
			inputs[i] = 1
		} else if gs.board[i] == "O" {
			inputs[i] = -1
		}
	}
	return inputs
}

func (gs *GameState) checkGameOver(winner *string) bool {
	for i := 0; i < 3; i++ {
		if gs.board[i*3] == gs.board[i*3+1] && gs.board[i*3] == gs.board[i*3+2] && gs.board[i*3] != "." {
			*winner = gs.board[i*3]
			return true
		}
		if gs.board[i] == gs.board[i+3] && gs.board[i] == gs.board[i+6] && gs.board[i] != "." {
			*winner = gs.board[i]
			return true
		}
	}

	for i := 0; i < 3; i++ {
		if gs.board[i] == gs.board[i+3] && gs.board[i] == gs.board[i+6] && gs.board[i] != "." {
			return true
		}
	}

	if gs.board[0] == gs.board[4] && gs.board[0] == gs.board[8] && gs.board[0] != "." {
		*winner = gs.board[0]
		return true
	}
	if gs.board[2] == gs.board[4] && gs.board[2] == gs.board[6] && gs.board[2] != "." {
		*winner = gs.board[2]
		return true
	}

	emptyTiles := 0
	for i := 0; i < 9; i++ {
		if gs.board[i] == "." {
			emptyTiles++
		}
	}
	if emptyTiles == 0 {
		*winner = "draw"
		return true
	}

	return false
}

func getComputerMove(gs *GameState, nn *NeuralNetwork, displayProbas bool) int {

	inputs := gs.boardToInputs()
	nn.forwardPass(inputs)

	maxProba := -1.0
	maxProbaIdx := -1
	bestMove := -1
	bestLegalProba := -1.0

	for i, proba := range nn.outputs {
		if proba > maxProba {
			maxProba = proba
			maxProbaIdx = i
		}

		if gs.board[i] == "." {
			if bestMove == -1 || proba > bestLegalProba {
				bestMove = i
				bestLegalProba = proba
			}
		}
	}

	if displayProbas {
		fmt.Println(nn.outputs)

		totalProba := 0.0

		for _, proba := range nn.outputs {
			totalProba += proba
		}

		fmt.Println(totalProba)
	}

	return maxProbaIdx
}

func (nn *NeuralNetwork) backwardPass(target_probas []float64, lr float64, rewardScale float64) {
	outputDelta := make([]float64, nn.outputSize)
	hiddenDelta := make([]float64, nn.hiddenSize)

	for i := 0; i < nn.outputSize; i++ {
		outputDelta[i] = (target_probas[i] - nn.outputs[i]) * math.Abs(rewardScale)
	}

	for i := 0; i < nn.hiddenSize; i++ {
		error := 0.0
		for j := 0; j < nn.outputSize; j++ {
			error += outputDelta[j] * nn.weightsHO[i*nn.outputSize+j]
		}

		hiddenDelta[i] = error * reLUDerivative(nn.hidden[i])
	}

	for i := 0; i < nn.hiddenSize; i++ {
		for j := 0; j < nn.outputSize; j++ {
			nn.weightsHO[i*nn.outputSize+j] -= lr * outputDelta[j] * nn.hidden[i]
		}
	}
	for j := 0; j < nn.outputSize; j++ {
		nn.biasesO[j] -= lr * outputDelta[j]	
	}

	for i := 0; i < nn.inputSize; i++ {
		for j := 0; j < nn.hiddenSize; j++ {
			nn.weightsIH[i*nn.hiddenSize+j] -= lr * hiddenDelta[j] * nn.inputs[i]
		}
	}
	for j := 0; j < nn.hiddenSize; j++ {
		nn.biasesH[j] -= lr * hiddenDelta[j]
	}
}

func learnFromGame(nn *NeuralNetwork, moveHistory []int, numMoves int, nnMovesEven bool, winner string) {
	var reward float64
	var nnSymbol string
	if nnMovesEven {
		nnSymbol = "O"
	} else {
		nnSymbol = "X"
	}

	if winner == "draw" {
		reward = 0.3
	} else if winner == nnSymbol {
		reward = 1.0
	} else {
		reward = -2.0
	}

	var gs GameState
	targetProbas := make([]float64, nn.outputSize)


}
