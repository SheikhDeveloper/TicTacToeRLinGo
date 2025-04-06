package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
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
		inputs: make([]float64, inputSize),
		hidden: make([]float64, hiddenSize),
		outputs: make([]float64, outputSize),
		rawLogits: make([]float64, outputSize),
	}
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
	maximum := (*input)[0]
	for _, v := range *input {
		if v > maximum {
			maximum = v
		}
	}
	var sum float64
	sum = 0
	for i, _ := range *output {
		(*output)[i] = math.Exp((*input)[i] - maximum)
		sum += (*output)[i]
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
	// copy input
	nn.inputs = input

	// Input to hidden
	for i := 0; i < nn.hiddenSize; i++ {
		sum := nn.biasesH[i]
		for j := 0; j < nn.inputSize; j++ {
			sum += nn.inputs[j] * nn.weightsIH[j*nn.hiddenSize+i]
		}
		nn.hidden[i] = reLU(sum)
	}

	// Hidden to output
	for i := 0; i < nn.outputSize; i++ {
		nn.rawLogits[i] = nn.biasesO[i]
		for j := 0; j < nn.hiddenSize; j++ {
			nn.rawLogits[i] += nn.hidden[j] * nn.weightsHO[j*nn.outputSize+i]
		}
	}

	softmax(&nn.rawLogits, &nn.outputs)
}

func (gs *GameState) initGame() {
	gs.currentPlayer = 0
	gs.board = [9]string{"." , "." , "." , "." , "." , "." , "." , "." , "."}
} 

func displayBoard(board [9]string) {
	fmt.Println(board[0], board[1], board[2])
	fmt.Println(board[3], board[4], board[5])
	fmt.Println(board[6], board[7], board[8])
}

func (gs *GameState) boardToInputs(inputs []float64) []float64 {
	for i := 0; i < 9; i++ {
		if gs.board[i] == "." {
			inputs[i * 2] = 0
			inputs[i * 2 + 1] = 0
		} else if gs.board[i] == "X" {
			inputs[i * 2] = 1
			inputs[i * 2 + 1] = 0
		} else {
			inputs[i * 2] = 0
			inputs[i * 2 + 1] = 1
		}
	}
	return inputs
}

func (gs *GameState) checkGameOver(winner *string) bool {
	for i := 0; i < 3; i++ {
		if gs.board[i*3] != "." && gs.board[i*3] == gs.board[i*3+1] && gs.board[i*3+1] == gs.board[i*3+2] {
			*winner = gs.board[i*3]
			return true
		}
	}

	for i := 0; i < 3; i++ {
		if gs.board[i] == gs.board[i+3] && gs.board[i+3] == gs.board[i+6] && gs.board[i] != "." {
			*winner = gs.board[i]
			return true
		}
	}

	if gs.board[0] == gs.board[4] && gs.board[4] == gs.board[8] && gs.board[0] != "." {
		*winner = gs.board[0]
		return true
	}
	if gs.board[2] == gs.board[4] && gs.board[4] == gs.board[6] && gs.board[2] != "." {
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

	inputs := make([]float64, nn.inputSize)
	inputs = gs.boardToInputs(inputs)
	nn.forwardPass(inputs)

	maxProba := -1.0
	bestMove := -1
	bestLegalProba := -1.0

	for i, proba := range nn.outputs {
		if proba > maxProba {
			maxProba = proba
		}

		if gs.board[i] == "." && (bestMove == -1 || proba > bestLegalProba) {
				bestMove = i
				bestLegalProba = proba
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

	return bestMove
}

func (nn *NeuralNetwork) backwardPass(target_probas []float64, lr float64, rewardScale float64) {
	outputDelta := make([]float64, nn.outputSize)
	hiddenDelta := make([]float64, nn.hiddenSize)

	for i := 0; i < nn.outputSize; i++ {
		outputDelta[i] = (nn.outputs[i] - target_probas[i]) * math.Abs(rewardScale)
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

func (nn *NeuralNetwork) learnFromGame(moveHistory []int, numMoves int, nnMovesEven bool, winner string) {
	var reward float64
	var nnSymbol string
	if nnMovesEven {
		nnSymbol = "O"
	} else {
		nnSymbol = "X"
	}

	if winner == "draw" {
		reward = 0.3 // small reward for draw
	} else if winner == nnSymbol {
		reward = 1.0 // big reward for winning
	} else {
		reward = -2.0 // negative reward for losing
	}

	var gs GameState
	targetProbas := make([]float64, nn.outputSize)

	for moveIdx := 0; moveIdx < numMoves; moveIdx++ {
		if (nnMovesEven && (moveIdx % 2) == 0) || (!nnMovesEven && (moveIdx % 2) == 1) {
			continue
		}

		gs.initGame()
		for i := 0; i < moveIdx; i++ {
			var symbol string 
			if i % 2 == 0 {
				symbol = "X"
			} else {
				symbol = "O"
			}
			gs.board[moveHistory[i]] = symbol
		}

		inputs := make([]float64, nn.inputSize)
		inputs = gs.boardToInputs(inputs)

		nn.forwardPass(inputs)

		move := moveHistory[moveIdx]

		move_importance := 0.5 + 0.5 * float64(moveIdx) / float64(numMoves)
		scaledReward := reward * move_importance

		for i := 0; i < nn.outputSize; i++ {
			targetProbas[i] = 0
		}

		if scaledReward >= 0 {
			targetProbas[move] = 1.0
		} else {
			validMovesLeft := 9 - moveIdx - 1
			otherProba := 1.0 / float64(validMovesLeft)
			for i := 0; i < 9; i++ {
				if gs.board[i] == "." && i != move {
					targetProbas[i] = otherProba
				}
			}
		}

		nn.backwardPass(targetProbas, 0.01, scaledReward)
	}
}

func (nn *NeuralNetwork) playGame() {
	var gs GameState
	var winner string
	moveHistory := make([]int, 9)
	numMoves := 0

	gs.initGame()

	fmt.Println("Welcome to Tic Tac Toe! You are X, the computer is O.")
	fmt.Println("Enter positions as numbers from 0 to 8 (see picture).")

	for !gs.checkGameOver(&winner) {
		displayBoard(gs.board)

		if gs.currentPlayer == 0 {
			
			var move int
			var movec string
			fmt.Println("Your move (0-8): ")

			fmt.Scan(&movec)
			move, _ = strconv.Atoi(movec)

			if move < 0 || move > 8 || gs.board[move] != "." {
				fmt.Println("Invalid move! Try again.")
				continue
			}

			gs.board[move] = "X"
			moveHistory[numMoves] = move
			numMoves++
		} else {
			fmt.Println("Computer's move:")
			move := getComputerMove(&gs, nn, true)
			gs.board[move] = "O"
			fmt.Println("Computer placed O in position", move)
			moveHistory[numMoves] = move
			numMoves++
		}

		gs.currentPlayer = gs.currentPlayer ^ 1
	}

	displayBoard(gs.board)
	if winner == "draw" {
		fmt.Println("Draw!")
	} else if winner == "X" {
		fmt.Println("You win!")
	} else {
		fmt.Println("Computer wins!")
	}

	nn.learnFromGame(moveHistory, numMoves, true, winner)
}

func (gs *GameState) getRandomMove() int {
	for {
		move := rand.Intn(9)
		if gs.board[move] != "." {
			continue
		}
		return move
	}
}

func (nn NeuralNetwork) playRandomGame(moveHistory []int, numMoves *int) string {
	var gs GameState
	var winner string
	*numMoves = 0

	gs.initGame()

	for !gs.checkGameOver(&winner) {
		var move int

		if gs.currentPlayer == 0 {
			move = gs.getRandomMove()
		} else {
			move = getComputerMove(&gs, &nn, false)
		}

		var symbol string
		if gs.currentPlayer == 0 {
			symbol = "X"
		} else {
			symbol = "O"
		}
		gs.board[move] = symbol
		moveHistory[*numMoves] = move
		*numMoves++

		gs.currentPlayer = gs.currentPlayer ^ 1
	}

	nn.learnFromGame(moveHistory, *numMoves, true, winner)
	return winner
}

func (nn NeuralNetwork) trainAgainstRandom(numGames int) {
	moveHistory := make([]int, 9)
	numMoves := 0
	wins := 0
	losses := 0
	ties := 0

	fmt.Printf("Training NN against %d random games...\n", numGames)

	playedGames := 0
	for i := 0; i < numGames; i++ {
		winner := nn.playRandomGame(moveHistory, &numMoves)
		playedGames++

		if winner == "draw" {
			ties++
		} else if winner == "O" {
			wins++
		} else {
			losses++
		}

		if (i + 1) % 10000 == 0 {
			fmt.Printf("Played %d games. Wins: %.2f%%, Losses: %.2f%%, Ties: %.2f%%\n", playedGames, float64(wins) / float64(playedGames) * 100, 
			float64(losses) / float64(playedGames) * 100, float64(ties) / float64(playedGames) * 100)
			playedGames = 0
			wins = 0
			losses = 0
			ties = 0
		}
	}

	fmt.Println("Training complete!")
}

func main() {

	randomGames := 150000

	nn := initNN(18, 9, 512)

	if randomGames > 0 { nn.trainAgainstRandom(randomGames) }

	for {
		var playAgain string
		nn.playGame()

		fmt.Println("Play again? (y/n)")
		fmt.Scan(&playAgain)
		if playAgain != "y" && playAgain != "Y" {
			break
		}
	}
}
