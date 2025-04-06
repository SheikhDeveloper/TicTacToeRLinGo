package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
)

/* Game state */
type GameState struct {
	board [9]string // board as an array
	currentPlayer int // 0 for X(player), 1 for O (computer)
}

/* Neural network */
type NeuralNetwork struct {
	inputSize int
	outputSize int
	hiddenSize int
	weightsIH []float64 // input to hidden weights
	weightsHO []float64 // hidden to output weights
	biasesH []float64 // hidden biases
	biasesO []float64 // output biases

	inputs []float64
	hidden []float64
	rawLogits []float64 // logits before applying softmax
	outputs []float64
}

/* Create new neural network
 * @param inputSize the size of the input
 * @param outputSize the size of the output
 * @param hiddenSize the size of the hidden layer
 */
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

/* ReLU
 * @param x the input
 * @return the output
 */
func reLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

/* Derivative of ReLU
 * @param x the input
 * @return the derivative
 */
func reLUDerivative(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

/* Initialize neural network with random weights
 * @param inputSize the size of the input
 * @param outputSize the size of the output
 * @param hiddenSize the size of the hidden layer
 */
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

/* Softmax function
 * @param input the input logits
 * @param output the output probabilities
 */
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

/* Forward pass of NN
 * @param input the input
 */
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

/* Init game
 * @param gs game state
 */
func (gs *GameState) initGame() {
	gs.currentPlayer = 0
	gs.board = [9]string{"." , "." , "." , "." , "." , "." , "." , "." , "."}
} 

/* Display board
 * @param board as an array
 */
func displayBoard(board [9]string) {
	fmt.Println(board[0], board[1], board[2])
	fmt.Println(board[3], board[4], board[5])
	fmt.Println(board[6], board[7], board[8])
}

/* Convert board to inputs
 * @param gs game state
 * @param inputs empty container to store inputs
 * @return inputs
 */
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

/* Check whether the terminating condition is met
 * @param gs game state
 * @param winner pointer to string (winner of the game)
 * @return whether the game is over
 */
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

/* Get computer move
 * @param gs game state
 * @param nn neural network
 * @param displayProbas bool. Purely for debug
 */
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

	if displayProbas { // logs purely for debug. Can be removed by setting displayProbas to false
		fmt.Println(nn.outputs)

		totalProba := 0.0

		for _, proba := range nn.outputs {
			totalProba += proba
		}

		fmt.Println(totalProba)
	}

	return bestMove
}

/* Backward pass of NN 
 * @param target_probas the target probabilities of the game
 * @param lr the learning rate
 * @param rewardScale the reward scale
 */
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

/* 
 * Function for NN to learn from a game
 * moveHistory is the list of moves played in the game so far
 * numMoves is the number of moves played in the game so far
 * nnMovesEven is true if the NN is playing as X, false if it is playing as O
*/
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

/*
 * Function to play a game against the trained computer
 * @param nn the trained neural network
 */

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
			move := getComputerMove(&gs, nn, false) // setting displayProbas to true logs the probas for debug. Can be removed by setting displayProbas to false
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

/* 
    Make a random move
	Used for training against random games
	@param gs game state
	@return move (place on board)
*/

func (gs *GameState) getRandomMove() int {
	for {
		move := rand.Intn(9)
		if gs.board[move] != "." {
			continue
		}
		return move
	}
}

/* 
    Play a random game against the computer
	@param moveHistory container to store move history
	@param numMoves number of moves played
	@return winner
*/

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

/*
	Training against random games
	@param numGames number of games to play
*/
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
			fmt.Printf("Played %d games. In the last 10000: Wins: %.2f%%, Losses: %.2f%%, Ties: %.2f%%\n", i + 1, float64(wins) / float64(playedGames) * 100, 
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
