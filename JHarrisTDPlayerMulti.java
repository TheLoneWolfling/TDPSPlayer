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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

// Basic idea:
// I have an evolved function estimating the mean value of a particular
// play.
// I run a modified version of MCTS over the possible plays for a given
// board state - modified by greedily using the estimator instead of playing
// randomly
// i.e. shuffle cards remaining, then run a single game with that set of
// cards
// for all possible plays. Repeat until time is up.
// Then combine a weighted sum of said MCTS average result and the estimator directly
// to pick a final value.

// Function estimator was evolved by a variant on TD learning:
// Play a game through, then update all estimates for played
// board states towards the final estimated value, using an exponentially
// decaying learning rate the earlier in the game you get.

// I started with naive TD learning, but for various reasons it tended to be
// very unstable

public class JHarrisTDPlayerMulti implements PokerSquaresPlayer {

	// How long (in ms) to leave as a buffer against overshooting the time limit
	private static final long MILLIS_TIME_BUFFER = 500;

	// How long (in ms) to leave to do final move selection
	// Realistically, not even needed.
	private static final long MILLIS_TIME_FINAL = 2;
	
	// 0.0 = only use MCTS result
	// 1.0 = only use BPANNE result
	// intermediate values lerp between the two.
	private static final double ESTIMATION_WEIGHT = 0.2;
	
	// How many threads to use.
	private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

	private static final JHarrisBPANNE estimator = loadEstimatorFromFile("JHarrisTDEstimator.dat");

	private static final ExecutorService pool = Executors.newFixedThreadPool(NUM_THREADS);

	private static final CompletionService<long[][]> completionService = new ExecutorCompletionService<long[][]>(pool);

	public static final boolean ASSERTIONS_ENABLED = checkAssertionsEnabled();


	// Number of cards played on the board so far.
	// Note: in the middle of getPlay it's generally one off, as is
	// numCardsRemaining
	private /* unsigned */ int numCardsPlayed;

	// Note: number of cards remaining *to play*, not number of cards in the
	// deck.
	// So: 0-25, not 0-52.
	private /* unsigned */ int numCardsRemaining;

	// Note: last `numCardsPlayed` cards are the cards already played
	// And the first play is card #0, the second card #1, and so on for the
	// MCTS.
	private Card[] cardsRemainingInDeck;

	// XY coordinates of the remaining free positions on the board.
	// Note that this gets "shuffled" around - the last `numCardsPlayed`
	// positions are
	// the positions already played.
	private int[][] freePositions;

	private Card[][] board;

	private static final PokerSquaresPointSystem POINT_SYSTEM = PokerSquaresPointSystem.getAmericanPointSystem();

	// Should be 1000
	private static final int ESTIMATOR_SCALE = normalizeMinMax(POINT_SYSTEM.getScoreTable());

	private static int normalizeMinMax(int[] pointTable) {
		// Scales points so they are within [-1,1], inclusive
		int min = Integer.MAX_VALUE;
		int max = Integer.MIN_VALUE;
		for (int entry : pointTable) {
			min = Math.min(min, entry);
			max = Math.max(max, entry);
		}
		final int numRowsAndCols = 5 + 5;
		return Math.max(Math.abs(min), Math.abs(max)) * numRowsAndCols;
	}

	private static JHarrisBPANNE loadEstimatorFromFile(String filename) {
		// Blehh...

		FileInputStream fileIn = null;
		ObjectInputStream in = null;
		try {
			fileIn = new FileInputStream(filename);
			in = new ObjectInputStream(fileIn);
			return (JHarrisBPANNE) in.readObject();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
			throw new RuntimeException("Cannot find estimator?");
		} finally {
			try {
				if (in != null)
					in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			try {
				if (fileIn != null)
					fileIn.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public JHarrisTDPlayerMulti() {
		System.out.println(getName() + " using " + NUM_THREADS + " threads.");
	}

	@Override
	public void setPointSystem(PokerSquaresPointSystem system, long millis) {
		// This method should not do anything.

		assert(Arrays.equals(system.getScoreTable(), POINT_SYSTEM.getScoreTable()));
	}

	@Override
	public void init() {
		// Called before each game

		this.numCardsPlayed = 0;
		this.numCardsRemaining = 25;
		this.cardsRemainingInDeck = Card.getAllCards();
		this.freePositions = makeFreePositions();
		this.board = makeBoard();
	}

	@Override
	public int[] getPlay(Card card, long millisRemaining) {
		long startTime = System.currentTimeMillis();

		doSanityCheck();

		// This is relatively expensive, but boo hoo.
		// Relatively expensive, but done 25 times / 30 seconds means that
		// it's not worth it to try to deal with currently.
		int index = getIndexOfCardInDeck(card, cardsRemainingInDeck);

		assert(cardsRemainingInDeck[index].getCardId() == card.getCardId());

		swap(cardsRemainingInDeck, index, 51 - numCardsPlayed);

		// Should always be overwritten, but best to keep something sane
		// here for the off chance that we're *really* tight on time.
		int play = numCardsRemaining - 1;

		// Previously I ignored the first play, but due to the way
		// the function estimator focuses on certain parts of the game tree
		// it does (substantially) better if I let it "waste" the time figuring
		// out where to play the first card. Go figure.
		
		if (numCardsRemaining > 1) {

			long millisPerPlayAndCleanup = (long) ((millisRemaining - MILLIS_TIME_BUFFER) / (double) (numCardsRemaining - 1));
	
			long millisForPlay = millisPerPlayAndCleanup - MILLIS_TIME_FINAL;
	
			long endTime = startTime + millisForPlay;
	
			// The one annoying thing about doing this as a do/undo
			// setup as opposed to doing (slow) defensive copies everywhere:
			// So many "oops I just trashed the state of the board" errors.
	
			Card[][] temp = null;
			if (ASSERTIONS_ENABLED) {
				temp = board.clone();
				for (int ii = 0; ii < 5; ii++)
					temp[ii] = board[ii].clone();
			}
			
			MonteCarloCallable[] tasks = new MonteCarloCallable[NUM_THREADS];
			for (int i = 0; i < NUM_THREADS; i++) {
				tasks[i] = new MonteCarloCallable(this, card, endTime); // Wish I didn't have to do all this constructing. Oh well.
			}
			for (int i = 0; i < NUM_THREADS; i++) {
				completionService.submit(tasks[i]);
			}
			// While tasks are running, the main thread will steal enough cycles to
			// calculate the estimated values of this move directly.
			double[] estimatedValues = new double[numCardsRemaining];
			for (int i = 0; i < numCardsRemaining; i++) {
				doBoardPlay(board, card, freePositions[i]);
				estimatedValues[i] = getValueForBoardPos(board);
				undoBoardPlay(board, freePositions[i]);
				assert(Arrays.deepEquals(board, temp));
			}
			
			// Now as tasks return sum the counts and sums...
	
			long[] monteCarloSums = new long[numCardsRemaining];
			long[] monteCarloCounts = new long[numCardsRemaining];
			for (int i = 0; i < NUM_THREADS; i++) {
				try {
					long[][] comp = completionService.take().get(); // Acts as a memory
																// barrier, yay!
					for (int j = 0; j < numCardsRemaining; j++) {
						monteCarloCounts[j] += comp[0][j];
						monteCarloSums[j] += comp[1][j];
					}
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
					continue; // Better to ignore it than to suffer a zero.
				}
			}
	
			// Find best value
	
			double bestValue = Double.NEGATIVE_INFINITY;
	
			long count = 0;
			for (int j = 0; j < numCardsRemaining; j++) {
				count += monteCarloCounts[j];
				final double value;
				if (monteCarloCounts[j] == 0 || ESTIMATION_WEIGHT == 1) // Only happens if we're *really*
												// pressed for time...
					value = estimatedValues[j];
				else
					value = estimatedValues[j] * ESTIMATION_WEIGHT
							+ monteCarloSums[j] / (double) monteCarloCounts[j] * (1 - ESTIMATION_WEIGHT);
				// Had fancy tie-breaking, but realized that it never came up due to
				// the
				// estimated values being ~unique.
				if (value > bestValue) {
					play = j;
					bestValue = value;
				}
	
			}
			// Debug:
			System.out.println(count + "\t" + monteCarloSums[play] / (double) monteCarloCounts[play] + "\t" + estimatedValues[play] + "\t" + bestValue + "\t" + (System.currentTimeMillis() - endTime));
		}

		int[] xyPlay = freePositions[play];

		numCardsPlayed += 1;
		numCardsRemaining -= 1;

		doBoardPlayAndFreePos(freePositions, board, card, play, numCardsRemaining);
		// No need for defensive copy here...
		// return xyPlay.clone();
		return xyPlay;
	}

	public static boolean checkAssertionsEnabled() {
		boolean checkAssert = false;
		assert checkAssert = true; // Side effect is intentional
		if (checkAssert)
			System.out.println("Assertions are enabled");
		return checkAssert;
	}

	private static double getValueForBoardPos(Card[][] board) {
		double[] input = new double[25 * (1 + 1 + 1) + 1];
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
		int boardScore = POINT_SYSTEM.getScore(board);
		if (numFree == 0)
			return boardScore;
		input[ind++] = boardScore / (double) ESTIMATOR_SCALE;

		assert(ind == input.length);

		return estimator.doEstimate(input) * (double) ESTIMATOR_SCALE;
		
	}

	private static void undoBoardPlay(Card[][] board, int[] pos) {
		assert(board[pos[0]][pos[1]] != null);

		board[pos[0]][pos[1]] = null;
	}

	private static void doBoardPlay(Card[][] board, Card card, int[] pos) {
		assert(board[pos[0]][pos[1]] == null);

		board[pos[0]][pos[1]] = card;
	}

	private static int getValueOfGamePlayedToEnd(Card[][] board, Card[] cardsRemainingInDeck, int[][] freePositions,
			int numCardsLeft, int numCardsPlayed) {
		// Recursive

		// Base case:
		if (numCardsLeft == 0)
			return POINT_SYSTEM.getScore(board);

		Card[][] temp = null;
		if (ASSERTIONS_ENABLED) {
			temp = board.clone();
			for (int ii = 0; ii < 5; ii++)
				temp[ii] = board[ii].clone();
		}

		int bestIndex = 0;
		double best = Double.NEGATIVE_INFINITY;
		Card card = cardsRemainingInDeck[numCardsPlayed];
		for (int j = 0; j < numCardsLeft; j++) {
			doBoardPlay(board, card, freePositions[j]);
			double value = getValueForBoardPos(board);
			undoBoardPlay(board, freePositions[j]);

			assert(Arrays.deepEquals(board, temp));

			// Due to the estimator ties appear so rarely that it's not worth
			// accounting for them.
			if (value > best) {
				bestIndex = j;
				best = value;
			}
		}
		assert(Arrays.deepEquals(board, temp));

		doBoardPlayAndFreePos(freePositions, board, card, bestIndex, numCardsLeft - 1);
		// recurse!
		int toRet = getValueOfGamePlayedToEnd(board, cardsRemainingInDeck, freePositions, numCardsLeft - 1,
				numCardsPlayed + 1);
		undoBoardPlayAndFreePos(freePositions, board, bestIndex, numCardsLeft - 1);

		assert(Arrays.deepEquals(board, temp));

		return toRet;
	}

	private static void doBoardPlayAndFreePos(int[][] freePositions, Card[][] board, Card card, int index,
			int numCardsLeft) {
		doBoardPlay(board, card, freePositions[index]);
		swap(freePositions, index, numCardsLeft);
	}

	private static void undoBoardPlayAndFreePos(int[][] freePositions, Card[][] board, int index, int numCardsLeft) {
		swap(freePositions, index, numCardsLeft);
		undoBoardPlay(board, freePositions[index]);
	}

	private static <T> void swap(T[] array, int indA, int indB) {
		T temp = array[indA];
		array[indA] = array[indB];
		array[indB] = temp;
	}

	private static int getIndexOfCardInDeck(Card card, Card[] cardsRemainingInDeck) {
		int id = card.getCardId();
		for (int i = 0; i < cardsRemainingInDeck.length; i++)
			if (cardsRemainingInDeck[i].getCardId() == id)
				return i;
		assert(false);
		return 0;
		// Shouldn't happen, but if we're here it's better to potentially pick
		// any points
		// we *can* get.
	}

	private void doSanityCheck() {
		if (!ASSERTIONS_ENABLED)
			return;
		
		assert(MILLIS_TIME_BUFFER >= 0);
		
		assert (MILLIS_TIME_FINAL >= 0);
		
		assert (ESTIMATION_WEIGHT >= 0);
		assert (ESTIMATION_WEIGHT <= 1);
		
		assert (NUM_THREADS >= 1);

		// Ensure things that shouldn't be null, aren't
		assert(board != null);
		assert(cardsRemainingInDeck != null);
		assert(freePositions != null);

		// Ensure number of cards remaining is within limits
		assert(numCardsRemaining > 0);
		assert(numCardsRemaining <= 25);

		// Ensure number of cards played is within limits
		assert(numCardsPlayed >= 0);
		assert(numCardsPlayed < 25);

		// Ensure number of cards played matches number of cards remaining
		assert(25 == numCardsPlayed + numCardsRemaining);

		// Ensure the deck still contains the right number of cards...
		assert(cardsRemainingInDeck.length == 52);

		// ...and that every card in the deck is there at most once...
		boolean[] cardsFound = new boolean[52];
		for (Card c : cardsRemainingInDeck) {
			assert(c != null);
			assert(!cardsFound[c.getCardId()]);
			cardsFound[c.getCardId()] = true;
		}
		// ...and that every card in the deck is there at least once.
		for (boolean b : cardsFound)
			assert(b);

		// Ensure that every card on the board is there at most once...
		int count = 0;
		boolean[] cardsFoundOnBoard = new boolean[52];
		for (int x = 0; x < 5; x++) {
			assert(board[x] != null);
			for (int y = 0; y < 5; y++) {
				if (board[x][y] != null) {
					count += 1;
					assert(!cardsFoundOnBoard[board[x][y].getCardId()]);
					cardsFoundOnBoard[board[x][y].getCardId()] = true;
				}
			}
		}
		// ... and that the number of cards on the board matches the number of
		// cards played.
		assert(count == numCardsPlayed);

		boolean[][] bBoard = new boolean[5][];
		for (int x = 0; x < 5; x++) {
			for (int y = 0; y < 5; y++) {
				bBoard[x] = new boolean[5];
			}
		}

		// Ensure the free positions list is the correct length
		assert(freePositions.length == 25);

		// Ensure that every free position is in freePositions at most once
		for (int[] xy : freePositions) {
			assert(xy != null);
			assert(xy[0] >= 0);
			assert(xy[0] < 5);
			assert(xy[1] >= 0);
			assert(xy[1] < 5);
			assert(!bBoard[xy[0]][xy[1]]);
			bBoard[xy[0]][xy[1]] = true;
		}

		// Ensure that the free positions remaining are actually free spaces on
		// the board
		for (int i = 0; i < numCardsRemaining; i++) {
			int[] xy = freePositions[i];
			assert(board[xy[0]][xy[1]] == null);
		}

		// Ensure that every card that hasn't been played isn't on the board...
		for (int i = 0; i < 52 - numCardsPlayed; i++) {
			assert(!cardsFoundOnBoard[cardsRemainingInDeck[i].getCardId()]);
		}

		// ...and that every card that has been played is on the board
		for (int i = 52 - numCardsPlayed; i < 52; i++) {
			assert(cardsFoundOnBoard[cardsRemainingInDeck[i].getCardId()]);
		}

	}

	@Override
	public String getName() {
		return "James Harris' Multithreaded MCTS player, v2.9";
	}

	private static int[][] makeFreePositions() {
		int[][] freePositions = new int[25][];
		for (int i = 0; i < 25; i++)
			freePositions[i] = new int[] { i / 5, i % 5 };
		return freePositions;
	}

	private static Card[][] makeBoard() {
		Card[][] board = new Card[5][];
		for (int i = 0; i < 5; i++)
			board[i] = new Card[5];
		return board;
	}

	public static void main(String[] args) {
		long score = 0;
		for (long i = 0;; i++) { // optimistic!
			JHarrisTDPlayerMulti player = new JHarrisTDPlayerMulti();
			PokerSquares ps = new PokerSquares(player, POINT_SYSTEM);
			int playScore = ps.play();
			score += playScore;
			System.out.println((i + 1) + "\t" + playScore + "\t" + score / (i + 1.0));
		}
	}
	

	private static class MonteCarloCallable implements Callable<long[][]> {

		private Card[][] board;
		private double endTime;
		private int numCardsRemaining;
		private int numCardsPlayed;
		private Card[] cardsRemainingInDeck;
		private int[][] freePositions;
		private Card card;

		public MonteCarloCallable(JHarrisTDPlayerMulti play, Card card, double endTime) {
			// So many defensive copies :/
			this.freePositions = new int[25][];
			for (int j = 0; j < 25; j++)
				this.freePositions[j] = new int[] { play.freePositions[j][0], play.freePositions[j][1] };
			this.board = new Card[5][];
			for (int j = 0; j < 5; j++) {
				this.board[j] = new Card[5];
				for (int k = 0; k < 5; k++)
					this.board[j][k] = play.board[j][k];
			}
			this.numCardsRemaining = play.numCardsRemaining;
			this.numCardsPlayed = play.numCardsPlayed;

			this.endTime = endTime;
			this.card = card;

			this.cardsRemainingInDeck = play.cardsRemainingInDeck.clone();
		}

		@Override
		public long[][] call() {
			try {
				Card[][] temp = null;
				if (ASSERTIONS_ENABLED) {
					temp = board.clone();
					for (int ii = 0; ii < 5; ii++)
						temp[ii] = board[ii].clone();
				}
				long[] monteCarloCounts = new long[numCardsRemaining];
				long[] monteCarloSums = new long[numCardsRemaining];
				int i = numCardsRemaining;
				while (System.currentTimeMillis() < endTime) {
					if (i == numCardsRemaining) {
						i = 0;
						// Just shuffle the cards that haven't already been
						// played in the "real" game
						Collections.shuffle(Arrays.asList(cardsRemainingInDeck).subList(0, 51 - numCardsPlayed));
					}

					doBoardPlayAndFreePos(freePositions, board, card, i, numCardsRemaining - 1);
					int value = getValueOfGamePlayedToEnd(board, cardsRemainingInDeck, freePositions,
							numCardsRemaining - 1, numCardsPlayed + 1);
					monteCarloSums[i] += value;
					monteCarloCounts[i] += 1;
					undoBoardPlayAndFreePos(freePositions, board, i, numCardsRemaining - 1);
					assert(Arrays.deepEquals(board, temp));
					

					i += 1;
				}
				return new long[][] {monteCarloCounts, monteCarloSums};
			} catch (Exception ex) {
				System.err.println(ex);
				if (ASSERTIONS_ENABLED) {
					Thread t = Thread.currentThread();
					t.getUncaughtExceptionHandler().uncaughtException(t, ex);
					return null;
				}
				return new long[][] {new long[numCardsRemaining], new long[numCardsRemaining]};
			}
		}

	}

}
