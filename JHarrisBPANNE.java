/*
 * TDPSPlayer - a naive learned-function MCTS poker squares player
 * Copyright (C) 2016 James Harris
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This is a naive-backpropagation-based artificial neural network.
 * 
 * WISH: use a matrix library instead of explicit loops.
 * 
 * (The professor did not want any external libraries used,
 *  and it would be more complexity to write a matrix library
 *  and use it once than just to use explicit loops.)
 */

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

// BPANNE: BackPropagation Artifical Neural Network Estimator

// The version included here is "neutered" - it has had the learning infrastructure
// removed

public class JHarrisBPANNE implements Serializable {

	final PokerSquaresPointSystem pointSystem;

	private static final long serialVersionUID = 2L;

	final double[/* earlier layer */][/* earlier layer node */][/*
																 * later layer
																 * node
																 */] weights;

	final int[/* # layers */] topology; // # nodes in each layer - input layer
										// == layer 0. Includes bias nodes.

	// Should be 1000
	// WISH: change to offset and scale according to *actual* min/max scores
	// possible.
	// But there is no easy way to figure out either of those.
	final int estimatorScale;
	
	private JHarrisBPANNE(JHarrisBPANNE b) { // constructor for deserialization
        // copy the non-transient fields
		this.weights = b.weights;
		this.topology = b.topology;
		this.pointSystem = (b.pointSystem == null) ? PokerSquaresPointSystem.getAmericanPointSystem() : b.pointSystem;
		this.estimatorScale = (b.estimatorScale == 0) ? normalizeMinMax(pointSystem.getScoreTable()) : b.estimatorScale;
    }

    private Object readResolve() {
        // create a new object from the deserialized one
        return new JHarrisBPANNE(this);
    }

	private final int getNumLayers() {
		return topology.length;
	}

	private final int getOutputLayer() {
		return getNumLayers() - 1;
	}

	private final int getLayerSizeWithBias(int layer) {
		return topology[layer];
	}

	private final int getLayerSizeNoBias(int layer) {
		if (layer == getOutputLayer()) // Output node(s) have no bias node
										// as there is no next layer for the
										// bias node to connect to!
			return topology[layer];
		return topology[layer] - 1;
	}

	double doEstimate(double[] inputs) {

		// There are two ways to do bias values for neural network nodes
		// Either have separate bias values
		// Or have "nodes" that are never updated with a fixed activation of 1
		// I take the second route, as it is easier when one is not using a
		// matrix library.

		assert (inputs.length == getLayerSizeNoBias(0));

		final double[/* layer */][/* node */] nodeActivations = new double[getNumLayers()][];

		// If we were learning we'd hang on to nodeActivations for the backprop
		// phase.

		for (int i = 0; i < getNumLayers(); i++) {
			nodeActivations[i] = new double[getLayerSizeWithBias(i)];

			nodeActivations[i][getLayerSizeWithBias(i) - 1] = 1; // Bias nodes
																	// set to
																	// activation
																	// of 1
		}

		for (int i = 0; i < getLayerSizeNoBias(0); i++) { // Could use
															// System.arraycopy
															// here I suppose
			nodeActivations[0][i] = inputs[i]; // Frankly, I find this clearer.
			// Note that this leaves the bias node intact.
		}
		// For each layer...
		for (int nextLayer = 1; nextLayer < getNumLayers(); nextLayer++) {
			int prevLayer = nextLayer - 1;
			// For each node in the layer...
			for (int nextNode = 0; nextNode < getLayerSizeNoBias(nextLayer); nextNode++) {
				// Take a weighted sum of the inputs to the node...
				double sum = 0;
				for (int prevNode = 0; prevNode < getLayerSizeWithBias(nextLayer - 1); prevNode++) {
					sum += weights[prevLayer][prevNode][nextNode] * nodeActivations[prevLayer][prevNode];
				}
				// Run it through the activation function...
				nodeActivations[nextLayer][nextNode] = fwdFunc(sum);
				// And assign it to the activation of said node.
			}
		}
		// WISH: have a function be able to return, say double...
		// similar to varargs as the inputs of a function.
		return nodeActivations[nodeActivations.length - 1][0];
	}

	private double fwdFunc(double in) {
		// I tried other functions, notably
		// x / (abs(x) + 1)
		// They weren't enough faster to compensate for
		// them being slower.
		return Math.tanh(in);
	}

	@Override
	public String toString() {
		// TODO: change over to StringBuilder
		String toRet = "";
		toRet += "BPANNE[\n";
		for (int layer = 0; layer < getNumLayers(); layer++) {
			for (int prevNode = 0; prevNode < getLayerSizeWithBias(layer); prevNode++) {
				for (int nextNode = 0; nextNode < getLayerSizeWithBias(layer + 1); nextNode++) {
					toRet += weights[layer][prevNode][nextNode] + " ";
				}
				toRet += "\n";
			}
			toRet += "\n";
		}
		toRet += "]\n";
		return toRet;
	}

	public void saveToFile(String filename) throws IOException {
		saveToFile(this, filename);
	}

	/**
	 * Returns the normalization factor for a given hand point table
	 * 
	 * 
	 * Must be the case that -ESTIMATOR_SCALE <= minimum achievable score <=
	 * maximum achievable score <= ESTIMATOR_SCALE currently See ESTIMATOR_SCALE
	 * comment
	 * 
	 * Note that this must be the same between the BPANNE training run and the
	 * scoring run!
	 *
	 * IIRC, minimum / maximum actual scores are 0 and 725, respectively. WISH:
	 * do a training run with better scale and offset. So then scale would be
	 * ceil((725-0)/2) = 363, and offset would be 362 or 363. Probably 363, so
	 * the actual and predicted minimums coincide. TODO: pull into JHarrisBPANNE
	 *
	 * @param pointTable[10]
	 *            a table containing points for [high card, one pair, ...]
	 * @return scale an integer such that -scale <= min score <= max score <=
	 *         scale
	 */
	//
	static int normalizeMinMax(int[] pointTable) {
		// Scales points so they are within [-1,1], inclusive
		int min = Integer.MAX_VALUE;
		int max = Integer.MIN_VALUE;
		for (int entry : pointTable) {
			min = Math.min(min, entry);
			max = Math.max(max, entry);
		}
		final int numRowsAndCols = 5 + 5;
		// Currently assumes that offset=0, and so the achievable values are
		// between scale*-10 and scale*10
		return Math.max(Math.abs(min), Math.abs(max)) * numRowsAndCols;
	}

	public static void saveToFile(Serializable s, String filename) throws IOException {
		FileOutputStream fileOut = null;
		ObjectOutputStream out = null;
		try {
			fileOut = new FileOutputStream(filename);
			out = new ObjectOutputStream(fileOut);
			out.writeObject(s);
		} finally {
			if (out != null)
				out.close();
			if (fileOut != null)
				fileOut.close();
		}
	}

	@SuppressWarnings("unchecked")
	public static <T> T loadFromFile(String filename) throws IOException, ClassNotFoundException, ClassCastException {
		FileInputStream fileIn = null;
		ObjectInputStream in = null;
		try {
			fileIn = new FileInputStream(filename);
			in = new ObjectInputStream(fileIn);
			return (T) in.readObject();
		} finally {
			if (in != null)
				in.close();
			if (fileIn != null)
				fileIn.close();
		}
	}

	/**
	 * Gets the estimated mean score for a board.
	 * 
	 * Warning: this function cannot change behavior without a full training
	 * run. (Optimizations or sanity checks are fine; modifications of the
	 * inputs used are not.)
	 *
	 * @param Card[5][5]
	 *            board the board to estimate the score of
	 * @return the estimated mean score of the board
	 */
	public double getValueForBoardPos(Card[/* 5 */][/* 5 */] board) {

		JHarrisTDPlayerMulti.boardSanityCheck(board);

		// Current inputs for network:
		// isNull(x, y) -> 5x5=25
		// rank(x,y) -> 5x5=25
		// suit(x,y) -> 5x5=25
		// currentScore(board) -> 1=1

		// Some other possible inputs for the network:
		// cardId(x,y) -> 5x5=25
		// isSuit(x,y,suit) -> 5x5x4=100
		// isRank(x,y,rank) -> 5x5x13=650
		// minScore(rowOrCol) -> 5+5=10
		// maxScore(rowOrCol) -> 5+5=10
		// avgScoreGivenCardsRemaining(rowOrCol) -> 5+5=10
		// canBeScoredAs(rowOrCol, handType) -> (5+5)*10=100
		// numFree(board) -> 1=1
		// numPlayed(board) -> 1=1
		// minScore(board) -> 1=1
		// maxScore(board) -> 1=1
		// avgScore(board) -> 1=1
		// isCardPlayed(card) -> 52=52
		// isCardLeft(card) -> 52=52

		// With associated scaling / whitening?

		// WISH: use an auto-encoder to figure out optimum inputs then use said
		// auto-encoder encoder portion as the first couple layers of the
		// estimator.

		// WISH: run a meta-optimizer over the above to figure out optimum
		// inputs

		double[] input = new double[25 * (1 + 1 + 1) + 1];
		// Could be a List, but Java lack-of-optimizations makes it *slow*.
		// Especially with boxing / unboxing.

		int ind = 0;
		int numFree = 0;
		for (int i = 0; i < 25; i++) {
			Card c = board[i / 5][i % 5];
			input[ind++] = c == null ? 0 : 1;
			input[ind++] = c == null ? 0 : c.getSuit() / 4.0;
			input[ind++] = c == null ? 0 : c.getRank() / 13.0;
			if (c == null)
				numFree += 1;
		}
		int boardScore = pointSystem.getScore(board);
		if (numFree == 0)
			return boardScore;
		input[ind++] = boardScore / (double) estimatorScale;

		assert (ind == input.length);

		return doEstimate(input) * (double) estimatorScale;
	}

}
