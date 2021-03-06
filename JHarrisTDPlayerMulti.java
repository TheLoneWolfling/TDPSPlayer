
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
import java.io.FileNotFoundException;
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

// xxx: This shouldn't work / this is a bandaid patch.
// bug: Fix me please
// todo: I should look at this when I have time, but not urgent
// wish: If I had infinite time I'd do this, but I don't and my time is
//       probably better spent elsewhere. (Lower priority todo, give or take)

// Poker squares:
// You have a shuffled deck of cards and a 5x5 grid.
// Flip up the first card, place it at a free position on the grid.
// Repeat until all 25 cards are played.
// Then score each row / col as a 5-card poker hand (using the American point system, in this case).
// Sum said scores, and that is your final score.
// Minimum possible score is 0, maximum is ~725?

// American point system is broken; give or take, the optimum strategy is 
// "go for flushes in rows; go for whatever in columns" (or vice versa)
// with a few minor tweaks.

// The controller constructs an instance of this class,
// calls setPointSystem once (see comment for said method),
// calls init() before each game,
// then calls getPlay(<card>) 25x per game to get where to place the 25 cards.
// It then scores the game as above.
// It repeats init() through scoring for each game
// Then just takes the average final score.

// Note: the way the controller works any timeout / exception / invalid play is counted as a score of 0 for that game.
// As this is the minimum possible score, it is *never* worth it to throw an exception or time out.
// So there are a lot of places where exceptions are explicitly ignored in release mode.
// Yes, I had words with the professor about this.

// Basic idea for this player:
// I have an evolved function estimating the mean value of a particular
// play.
// I run a modified version of MCTS (Monte Carlo Tree Search) over the possible plays for a given
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
// very unstable. This is elaborated more in the design document, but suffice to say that
// naive TD learning tends to converge on always predicting the maximum points
// for all positions.

public class JHarrisTDPlayerMulti implements PokerSquaresPlayer {

	// How long (in ms) to leave as a buffer against overshooting the time limit
	private static final long MILLIS_TIME_BUFFER = 500;

	// How long (in ms) to leave to do final move selection
	// Realistically, not even needed.
	private static final long MILLIS_TIME_FINAL = 2;

	// 0.0 = only use MCTS result
	// 1.0 = only use BPANNE result
	// intermediate values lerp between the two.

	// Chosen by brute-force "try all values in [0,1] by a step of 0.1 and pick
	// best"
	// Probably different based on exact estimator
	// For the current estimator, score is pretty flat between ~0.2 to ~0.5 and
	// drops off above / below that.
	// WISH: use a meta-optimizer to find best value automagicially.
	private static final double ESTIMATION_WEIGHT = 0.2;

	// How many threads to use.
	// WISH: figure out optimum when including hyperthreading
	private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

	private static final ExecutorService pool = Executors.newFixedThreadPool(NUM_THREADS);

	private static final CompletionService<long[][]> completionService = new ExecutorCompletionService<long[][]>(pool);

	public static final boolean ASSERTIONS_ENABLED = JHarrisTDPlayerMulti.class.desiredAssertionStatus();
	
	/**
	 * Checks that the current static state is "sane", i.e. hasn't been
	 * corrupted
	 * 
	 * Doesn't do anything if assertions are not enabled.
	 */
	static void doStaticSanityChecks() {
		if (!ASSERTIONS_ENABLED)
			return;

		assert (MILLIS_TIME_BUFFER >= 0);
		assert (MILLIS_TIME_FINAL >= 0);

		assert (ESTIMATION_WEIGHT >= 0);
		assert (ESTIMATION_WEIGHT <= 1);

		assert (NUM_THREADS >= 1);
		
		assert (pool != null);
		assert (!pool.isTerminated());
		assert (!pool.isShutdown());
		
		assert (completionService != null);
	}
	
	static {
		doStaticSanityChecks();
	}

	private final JHarrisBPANNE estimator;

	// Number of cards played on the board so far.
	// Note: in the middle of getPlay it's generally one off, as is
	// numCardsRemaining
	private /* unsigned */ int numCardsPlayed; // 0 <= numCardsPlayed <= 25

	// Note: number of cards remaining *to play*, not number of cards in the
	// deck.
	// So: 0-25, not 0-52.
	// Note: in the middle of getPlay it's generally one off, as is
	// numCardsPlayed
	private /* unsigned */ int numCardsRemaining; // 0 <= numCardsRemaining <=
													// 25

	// Note for all of these arrays:
	// They would all be lists, except that Java is Java.
	// There's enough overhead that it does noticeably worse when these are
	// lists
	// as opposed to arrays.

	// Note: last `numCardsPlayed` cards are the cards already played
	// And the first play is card #0, the second card #1, and so on for the
	// MCTS.
	// Should not contain null.
	private Card[/* 52 */] cardsRemainingInDeck;

	// XY coordinates of the remaining free positions on the board.
	// Note that this gets "shuffled" around - the last `numCardsPlayed`
	// positions are the positions already played.
	// Should not contain null.
	private int[/* 25 */][/* 2 */] freePositions;

	// Null values mean "no card at position"
	// Indexed as board[x][y]
	// WISH: pull out to separate class with proper getter / setters
	// Probably overkill, though.
	private Card[/* 5 */][/* 5 */] board;

	public JHarrisTDPlayerMulti() throws ClassNotFoundException, ClassCastException, IOException {
		// WISH: log instead?
		System.out.println(getName() + " using " + NUM_THREADS + " threads.");

		estimator = JHarrisBPANNE.loadFromFile("JHarrisTDEstimator.dat");
	}

	/**
	 * @see PokerSquaresPlayer#setPointSystem(PokerSquaresPointSystem, long)
	 *      This method should be called before init is first called to set the
	 *      point system Note: only supports the American point system currently
	 * 
	 * @param system
	 *            The point system to use; must be the American point system
	 * @param millis
	 *            The timeout for doing any setup tasks at this time.
	 */
	@Override
	public void setPointSystem(PokerSquaresPointSystem system, long millis) {
		// We are writing a player to play with the American point system only.
		// But the driver supports parameterized point systems.
		// As such, this method should not do anything.

		// Check that it's setting it to the American point system only.
		assert (Arrays.equals(system.getScoreTable(), estimator.pointSystem.getScoreTable()));

		// ...and yet it does. Programming in a nutshell.
	}

	/**
	 * @see PokerSquaresPlayer#init() This method should be called before each
	 *      game played.
	 * 
	 */
	@Override
	public void init() {
		// Called before each game
		// A better name might be "reset()".

		this.numCardsPlayed = 0;
		this.numCardsRemaining = 25;
		// Overkill - could just re-sort the existing cards in deck
		// But this way I don't need to worry about errors compounding over
		// multiple games
		this.cardsRemainingInDeck = Card.getAllCards();
		// Ditto
		this.freePositions = makeFreePositions();
		// Ditto
		this.board = makeBoard();

		doSanityCheck();
	}

	/**
	 * @see PokerSquaresPlayer#getPlay(Card, long) This method should be called
	 *      25x per game with the cards flipped up before the first..25th plays.
	 * 
	 * @param card
	 *            The card flipped up
	 * @param millisRemaining
	 *            How many milliseconds remaining in the *game*.
	 * 
	 * @return xyPos={0<=x<5,0<=y<5} the position to play the card in
	 */
	@Override
	public int[/* 2 */] getPlay(Card card, long millisRemaining) {
		// Would move this inside the if block, below, but the sanity check
		// can take a long enough time that I don't want to risk timeouts.
		long startTime = System.currentTimeMillis();

		doSanityCheck();

		// Relatively expensive, but done 25 times / 30 seconds means that
		// it's not worth it to try to deal with currently. Boo hoo.
		// And by "relatively expensive" I mean it needs to do at max 52
		// lookups. Not too bad anyways.
		// Note: if cardsRemainingInDeck was a List, then I could just search it
		// via indexOf.
		// But Java is Java, and as such it's too slow if I do that.
		int index = getIndexOfCardInDeck(card, cardsRemainingInDeck);

		// What? You want the index of the card in the deck to actually be the
		// index of the card in the deck?
		// Preposterous!
		assert (cardsRemainingInDeck[index].getCardId() == card.getCardId());

		// As per the declaration: the cards played so far are swapped to the
		// back of the deck
		// This would be a micro-optimization, but it's done so often that it's
		// worth it.
		// As every Monte Carlo run involves playing all the way through to the
		// end then undoing it all...
		swap(cardsRemainingInDeck, index, 51 - numCardsPlayed);

		// Should always be overwritten, but best to keep something sane
		// here for the off chance that we're *really* tight on time.
		// Also used for the last card in each game
		int play = numCardsRemaining - 1;

		// Previously I ignored the first play, but due to the way
		// the function estimator focuses on certain parts of the game tree
		// it does (substantially) better if I let it "waste" the time figuring
		// out where to play the first card. Go figure.

		// WISH: run a training run with it set to only use "useful" positions
		// i.e. only one position per unique set of {row, column}. (That's a
		// set, so {row, column} == {column, row}.)
		// Should make it faster / better.
		// "Should".

		if (numCardsRemaining > 1) { // Last card played is trivial.

			// Calculate how long we have per play...

			long millisPerPlayAndCleanup = (long) ((millisRemaining - MILLIS_TIME_BUFFER)
					/ (double) (numCardsRemaining - 1));

			// ...Take off the time for doing final cleanup per play...

			long millisForPlay = millisPerPlayAndCleanup - MILLIS_TIME_FINAL;

			// ...and figure out our end time based on that.

			long endTime = startTime + millisForPlay;

			// The most annoying thing about doing this as a do/undo
			// setup as opposed to doing (slow) defensive copies everywhere:
			// So many "oops I just trashed the state of the board" errors.

			// So when debugging, make a defensive copy of the board to check
			// against later to ensure
			// that the board state doesn't get trashed.

			Card[][] temp = null;
			if (ASSERTIONS_ENABLED) {
				temp = board.clone();
				for (int ii = 0; ii < 5; ii++)
					temp[ii] = board[ii].clone();
			}

			// TODO: see if reusing MonteCarloCallables makes any sense at all.
			// It'd save NUM_THREADS allocations per play, which I suspect
			// wouldn't make
			// much of a difference. Nonetheless, worth a check.
			MonteCarloCallable[] tasks = new MonteCarloCallable[NUM_THREADS];
			for (int i = 0; i < NUM_THREADS; i++) {
				// I wish I didn't have to do all this constructing.
				// Oh well...
				tasks[i] = new MonteCarloCallable(this, card, endTime);
			}
			for (int i = 0; i < NUM_THREADS; i++) {
				completionService.submit(tasks[i]);
			}

			// While tasks are running, the main thread now steals enough cycles
			// to
			// calculate the estimated values of this move directly.
			double[] estimatedValues = new double[numCardsRemaining];
			for (int i = 0; i < numCardsRemaining; i++) {
				doBoardPlay(board, card, freePositions[i]);
				estimatedValues[i] = estimator.getValueForBoardPos(board);
				undoBoardPlay(card, board, freePositions[i]);
				assert (Arrays.deepEquals(board, temp));
			}

			// Now as tasks return sum the counts and sums...

			long[] monteCarloSums = new long[numCardsRemaining];
			long[] monteCarloCounts = new long[numCardsRemaining];
			for (int i = 0; i < NUM_THREADS; i++) {
				try {
					// Acts as a memory barrier, yay!
					long[][] comp = completionService.take().get();
					for (int j = 0; j < numCardsRemaining; j++) {
						monteCarloCounts[j] += comp[0][j];
						monteCarloSums[j] += comp[1][j];
					}
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace(); // WISH: Logging here
					continue; // Better to ignore it than to suffer a zero.
				}
			}

			// Find best value according to a weighted average of the directly
			// estimated value and the MC average value

			double bestValue = Double.NEGATIVE_INFINITY;

			long count = 0;
			for (int j = 0; j < numCardsRemaining; j++) {
				count += monteCarloCounts[j];
				final double value;
				if (monteCarloCounts[j] == 0 || ESTIMATION_WEIGHT == 1)
					// Only happens if we're *really* pressed for time...
					value = estimatedValues[j];
				else
					value = lerp(monteCarloSums[j] / (double) monteCarloCounts[j], estimatedValues[j], ESTIMATION_WEIGHT);

				// Had fancy tie-breaking, but realized that it never came up
				// due to the estimated values being ~unique.
				if (value > bestValue) {
					play = j;
					bestValue = value;
				}

			}
			// WISH: Proper logging
			System.out.println(count + "\t" + monteCarloSums[play] / (double) monteCarloCounts[play] + "\t"
					+ estimatedValues[play] + "\t" + bestValue + "\t" + (System.currentTimeMillis() - endTime));
		}

		int[] xyPlay = freePositions[play];

		numCardsPlayed += 1;
		numCardsRemaining -= 1;

		doBoardPlayAndFreePos(freePositions, board, card, play, numCardsRemaining);
		// No need for defensive copy here...
		// return xyPlay.clone();
		return xyPlay;
	}
	
	/**
	 * Lerp between a and b. sel=0 -> a, sel=1 -> b, intermediate values -> linear interpolation between a and b.
	 *
	 * @param a
	 * @param b
	 * @param sel
	 */
	public static double lerp(double a, double b, double sel) {
		assert sel >= 0;
		assert sel <= 1;
		
		double toRet = a * (1-sel) + b*(sel);
		
		assert toRet >= Math.min(a,  b);
		assert toRet <= Math.max(a,  b);
		return toRet;
	}

	/**
	 * Undoes a board play.
	 *
	 * @param Card[5][5] board 
	 * 				the board to undo the play on
	 * @param int[2] pos 
	 * 				the position to undo the play on
	 */
	private static void undoBoardPlay(Card old, Card[/* 5 */][/* 5 */] board, int[/* 2 */] pos) {

		boardSanityCheck(board);

		assert (pos != null);

		assert (pos.length == 2);

		assert (pos[0] >= 0);
		assert (pos[0] < 5);

		assert (pos[1] >= 0);
		assert (pos[1] < 5);

		assert (board[pos[0]][pos[1]] != null);

		assert (board[pos[0]][pos[1]] == old);

		board[pos[0]][pos[1]] = null;
	}

	/**
	 * Plays a card on a board.
	 *
	 * @param Card[5][5]
	 *            board the board to do the play on
	 * @param Card
	 *            card the card to play on the board
	 * @param int[2]
	 *            pos={0<=x<5, 0<=y<5} the position to play the card at
	 */
	private static void doBoardPlay(Card[/* 5 */][/* 5 */] board, Card card, int[/* 2 */] pos) {

		boardSanityCheck(board);

		assert (card != null);

		assert (pos != null);

		assert (pos.length == 2);

		assert (pos[0] >= 0);
		assert (pos[0] < 5);

		assert (pos[1] >= 0);
		assert (pos[1] < 5);

		assert (board[pos[0]][pos[1]] == null);

		board[pos[0]][pos[1]] = card;

		boardSanityCheck(board);
	}

	/**
	 * Gets the score of a game played from the passed-in board state to the
	 * end.
	 * @param estimator the estimator to use
	 *
	 * @param Card[5][5]
	 *            board the board to play the game on
	 * @param Card[52]
	 *            cardsRemainingInDeck the cards remaining in deck
	 * @param int[25][x,y]
	 *            freePositions see {@link JHarrisTDPlayerMulti.freePositions
	 *            freePositions}
	 * @param numCardsRemaining
	 *            How many cards are left to play
	 * @param numCardsPlayed
	 *            How many cards have been played
	 * @return the score of the game played to end
	 */
	private static int getValueOfGamePlayedToEnd(JHarrisBPANNE estimator, Card[/* 5 */][/* 5 */] board, Card[/* 52 */] cardsRemainingInDeck,
			int[/* 25 */][/* x,y */] freePositions, int numCardsRemaining, int numCardsPlayed) {
		// Recursive

		// Base case:
		if (numCardsRemaining == 0)
			return estimator.pointSystem.getScore(board);

		doSanityCheck(board, cardsRemainingInDeck, freePositions, numCardsRemaining, numCardsPlayed);

		// If assertions are enabled, make a copy of the board to check that the
		// do / undo setup is sound
		Card[][] temp = null;
		if (ASSERTIONS_ENABLED) {
			temp = board.clone();
			for (int ii = 0; ii < 5; ii++)
				temp[ii] = board[ii].clone();
		}

		Card card = cardsRemainingInDeck[51 - numCardsPlayed]; // Grab next card...

		int bestIndex = findbestPlayGreedy(estimator, board, freePositions, numCardsRemaining, temp, card);

		doBoardPlayAndFreePos(freePositions, board, card, bestIndex, numCardsRemaining - 1); // Do
		// the
		// best
		// play...

		int toRet = getValueOfGamePlayedToEnd(estimator, board, cardsRemainingInDeck, freePositions, numCardsRemaining - 1,
				numCardsPlayed + 1); // Recurse!

		
		undoBoardPlayAndFreePos(card, freePositions, board, bestIndex, numCardsRemaining - 1); // Undo
		// said
		// play...

		assert (Arrays.deepEquals(board, temp));// Check (yet again) that we
												// haven't messed anything up

		return toRet; // Return the value of the game played to the end
	}
	
	/**
	 * Gets the best position to play a card at according to an estimator.
	 * 
	 * @param estimator the estimator to use
	 *
	 * @param Card[5][5]
	 *            board the board to play the game on
	 * @param int[25][x,y]
	 *            freePositions see {@link JHarrisTDPlayerMulti.freePositions
	 *            freePositions}
	 * @param numCardsRemaining
	 *            How many cards are left to play
	 * @param Card[5][5]
	 *            boardCopy - only used if assertions are enabled; should be a clone of board
	 * @param Card card
	 * 				the card to find the best play for.
	 * @return the index of the best play in freePositions
	 */
	private static int findbestPlayGreedy(JHarrisBPANNE estimator, Card[][] board, int[][] freePositions,
			int numCardsRemaining, Card[][] boardCopy, Card card) {
		// WISH: there's a certain amount of duplication between this and getPlay
		// But getPlay has an array to fill, and this one doesn't.
		int bestIndex = 0;
		double best = Double.NEGATIVE_INFINITY;
		for (int j = 0; j < numCardsRemaining; j++) {
			doBoardPlay(board, card, freePositions[j]); // Try putting it in a
														// position...
			double value = estimator.getValueForBoardPos(board); // Get the estimated
														// value...
			undoBoardPlay(card, board, freePositions[j]); // Remove the card from said
													// position

			assert (Arrays.deepEquals(board, boardCopy)); // Check that we haven't
														// messed anything up

			// Due to the estimator ties appear so rarely that it's not even
			// worth
			// accounting for them.
			if (value > best) { // If it's the best we've see so far...
				bestIndex = j; // record it.
				best = value;
			}
		}
		assert (Arrays.deepEquals(board, boardCopy)); // Check (again) that we
													// haven't messed anything
													// up
		// Probably not needed.
		return bestIndex;
	}

	/**
	 * Plays a card on the board and manages the free position array.
	 * 
	 * @param int[25][x,y]
	 *            freePositions see {@link JHarrisTDPlayerMulti.freePositions
	 *            freePositions}
	 * @param Card[5][5]
	 *            board the board to play the game on
	 * @param card
	 *            the card to play
	 * @param index
	 *            the index of the play in the free positions array
	 * @param numCardsRemaining
	 *            How many cards are left to play
	 */
	private static void doBoardPlayAndFreePos(int[][] freePositions, Card[][] board, Card card, int index,
			int numCardsRemaining) {
		doBoardPlay(board, card, freePositions[index]); // Play the card to the
														// board...
		swap(freePositions, index, numCardsRemaining); // And record the play
														// position as no longer
														// free
	}

	/**
	 * Undoes a card play on the board and manages the free position array.
	 * 
	 * @param int[25][x,y]
	 *            freePositions see {@link JHarrisTDPlayerMulti.freePositions
	 *            freePositions}
	 * @param Card[5][5]
	 *            board the board to play the game on
	 * @param index
	 *            the index of the play in the free positions array
	 * @param numCardsRemaining
	 *            How many cards are left to play
	 */
	private static void undoBoardPlayAndFreePos(Card old, int[][] freePositions, Card[][] board, int index, int numCardsLeft) {
		swap(freePositions, index, numCardsLeft); // Remove the card from the
													// board...
		undoBoardPlay(old, board, freePositions[index]); // And record the play
													// position as free again
	}

	/**
	 * Swaps two indexes in an array
	 * 
	 * array[indA] <-> array[indB]
	 *
	 * @param <T>
	 *            The type of the array
	 * @param array
	 *            the array to swap indexes in
	 * @param indA
	 *            index a to swap
	 * @param indB
	 *            index b to swap
	 */
	private static <T> void swap(T[] array, int indA, int indB) { // Standard
																	// swap
																	// routine
																	// for
																	// arrays
		T temp = array[indA];
		array[indA] = array[indB];
		array[indB] = temp;
	}

	/**
	 * Gets the index of a card in a deck.
	 *
	 * @param card
	 *            the card to find in the deck
	 * @param cardsRemainingInDeck
	 *            the deck to find the card in
	 * @return the index of the card in the deck
	 */
	private static int getIndexOfCardInDeck(Card card, Card[] cardsRemainingInDeck) {

		for (int i = 0; i < cardsRemainingInDeck.length; i++) {
			assert ((card == cardsRemainingInDeck[i]) == card.equals(cardsRemainingInDeck[i]));
			if (card == cardsRemainingInDeck[i])
				return i;
		}
		
		assert (false);
		// Shouldn't happen, but if we're here it's better to potentially pick
		// any points we *can* get.
		return 0;
	}

	/**
	 * Checks that a board is "sane", i.e. hasn't been corrupted
	 * 
	 * Doesn't do anything if assertions are not enabled.
	 * 
	 * 
	 * TODO: there's a certain amount of duplication between this and
	 * doSanityCheck().
	 * 
	 * @param board
	 *            the board to check
	 */
	static void boardSanityCheck(Card[][] board) {
		if (!ASSERTIONS_ENABLED)
			return;

		assert (board != null);

		assert (board.length == 5);

		// Ensure that every card on the board is there at most once.
		boolean[] cardsFoundOnBoard = new boolean[52];
		for (int x = 0; x < 5; x++) {
			assert (board[x] != null);
			assert (board[x].length == 5);
			for (int y = 0; y < 5; y++) {
				if (board[x][y] != null) {
					assert (!cardsFoundOnBoard[board[x][y].getCardId()]);
					cardsFoundOnBoard[board[x][y].getCardId()] = true;
				}
			}
		}

	}

	/**
	 * Checks that the current state is "sane", i.e. hasn't been corrupted
	 * 
	 * Doesn't do anything if assertions are not enabled.
	 * 
	 */
	private void doSanityCheck() {
		if (!ASSERTIONS_ENABLED)
			return;

		doSanityCheck(board, cardsRemainingInDeck, freePositions, numCardsRemaining, numCardsPlayed);
	}

	/**
	 * Checks that the current state passed in is "sane", i.e. hasn't been
	 * corrupted
	 * 
	 * Doesn't do anything if assertions are not enabled.
	 * 
	 * @param Card[5][5]
	 *            board the board to check
	 * @param Card[52]
	 *            cardsRemainingInDeck the cards remaining in deck
	 * @param int[25][x,y]
	 *            freePositions see {@link JHarrisTDPlayerMulti.freePositions
	 *            freePositions}
	 * @param numCardsRemaining
	 *            How many cards are left to play
	 * @param numCardsPlayed
	 *            How many cards have been played
	 */
	private static void doSanityCheck(Card[][] board, Card[] cardsRemainingInDeck, int[][] freePositions,
			int numCardsRemaining, int numCardsPlayed) {
		if (!ASSERTIONS_ENABLED)
			return;
		
		doStaticSanityChecks();

		boardSanityCheck(board);

		// Ensure things that shouldn't be null, aren't
		assert (cardsRemainingInDeck != null);
		assert (freePositions != null);

		// Ensure number of cards remaining is within limits
		assert (numCardsRemaining > 0);
		assert (numCardsRemaining <= 25);

		// Ensure number of cards played is within limits
		assert (numCardsPlayed >= 0);
		assert (numCardsPlayed < 25);

		// Ensure number of cards played matches number of cards remaining
		assert (25 == numCardsPlayed + numCardsRemaining);

		// Ensure the deck still contains the right number of cards...
		assert (cardsRemainingInDeck.length == 52);

		// ...and that every card in the deck is there at most once...
		boolean[] cardsFound = new boolean[52];
		for (Card c : cardsRemainingInDeck) {
			assert (c != null);
			assert (!cardsFound[c.getCardId()]);
			cardsFound[c.getCardId()] = true;
		}
		// ...and that every card in the deck is there at least once.
		for (boolean b : cardsFound)
			assert (b);

		// Ensure that every card on the board is there at most once...
		int count = 0;
		boolean[] cardsFoundOnBoard = new boolean[52];
		for (int x = 0; x < 5; x++) {
			assert (board[x] != null);
			for (int y = 0; y < 5; y++) {
				if (board[x][y] != null) {
					count += 1;
					assert (!cardsFoundOnBoard[board[x][y].getCardId()]);
					cardsFoundOnBoard[board[x][y].getCardId()] = true;
				}
			}
		}
		// ... and that the number of cards on the board matches the number of
		// cards played.
		assert (count == numCardsPlayed);

		boolean[][] bBoard = new boolean[5][];
		for (int x = 0; x < 5; x++) {
			bBoard[x] = new boolean[5];
		}

		// Ensure the free positions list is the correct length
		assert (freePositions.length == 25);

		// Ensure that every free position is in freePositions at most once
		// And that the values in freePositions are "sane".
		for (int[] xy : freePositions) {
			assert (xy != null);
			assert (xy[0] >= 0);
			assert (xy[0] < 5);
			assert (xy[1] >= 0);
			assert (xy[1] < 5);
			assert (!bBoard[xy[0]][xy[1]]);
			bBoard[xy[0]][xy[1]] = true;
		}

		// Ensure that the free positions remaining are actually free spaces on
		// the board
		for (int i = 0; i < numCardsRemaining; i++) {
			int[] xy = freePositions[i];
			assert (board[xy[0]][xy[1]] == null);
		}

		// Ensure that every card that hasn't been played isn't on the board...
		for (int i = 0; i < 52 - numCardsPlayed; i++) {
			assert (!cardsFoundOnBoard[cardsRemainingInDeck[i].getCardId()]);
		}

		// ...and that every card that has been played is on the board
		for (int i = 52 - numCardsPlayed; i < 52; i++) {
			assert (cardsFoundOnBoard[cardsRemainingInDeck[i].getCardId()]);
		}

	}

	/**
	 * @see PokerSquaresPlayer#getName()
	 * 
	 * @return the name of this Poker Squares player.
	 */
	@Override
	public String getName() {
		return "James Harris' Multithreaded MCTS player, v3.0";
	}

	/**
	 * Makes a free position array.
	 *
	 * @return an int[25][0<=x<5,0<=y<5] representing all positions on the 5x5
	 *         board
	 */
	private static int[][] makeFreePositions() {
		int[][] freePositions = new int[25][];
		for (int i = 0; i < 25; i++)
			freePositions[i] = new int[] { i / 5, i % 5 };
		return freePositions;
	}

	/**
	 * Makes a blank board.
	 *
	 * @return a Card[5][5] filled with nulls representing a blank board.
	 */
	private static Card[][] makeBoard() {
		Card[][] board = new Card[5][];
		for (int i = 0; i < 5; i++)
			board[i] = new Card[5];
		return board;
	}

	/**
	 * The main method.
	 * 
	 * Runs "forever", printing out the score and mean score every so often.
	 * 
	 * @param args
	 *            not used
	 * @throws IOException 
	 * @throws ClassCastException 
	 * @throws ClassNotFoundException 
	 */
	public static void main(String[] args) throws ClassNotFoundException, ClassCastException, IOException {
		long score = 0;
		for (long i = 1;; i++) { // optimistic!
			JHarrisTDPlayerMulti player = new JHarrisTDPlayerMulti();
			PokerSquares ps = new PokerSquares(player, player.estimator.pointSystem);
			int playScore = ps.play();
			score += playScore;
			System.out.println(String.format("%16d%16d%16f", i, playScore, score / (double) i));
		}
	}

	private static class MonteCarloCallable implements Callable<long[][]> {
		// The below all have the same comments as JHarrisTDPlayerMulti
		private Card[][] board;
		private long endTime;
		private int numCardsRemaining;
		private int numCardsPlayed;
		private Card[] cardsRemainingInDeck;
		private int[][] freePositions;
		private JHarrisBPANNE estimator;

		// The card that we have to play currently.
		private Card card;

		/**
		 * Instantiates a new MonteCarloCallable.
		 * 
		 *
		 * @param play
		 *            the parent JHArrisTDPlayer
		 * @param card
		 *            the card that was flipped up
		 * @param endTime
		 *            the time that we should return at
		 */
		public MonteCarloCallable(JHarrisTDPlayerMulti play, Card card, long endTime) {
			// So many defensive copies :/
			// WISH: look at having the main player instead keep NUM_THREADS
			// copies of everything?
			
			play.doSanityCheck();

			this.freePositions = new int[25][];
			for (int j = 0; j < 25; j++)
				this.freePositions[j] = play.freePositions[j].clone();

			this.board = new Card[5][];
			for (int j = 0; j < 5; j++) {
				this.board[j] = play.board[j].clone();
			}

			this.numCardsRemaining = play.numCardsRemaining;
			this.numCardsPlayed = play.numCardsPlayed;

			this.endTime = endTime;
			this.card = card;

			this.cardsRemainingInDeck = play.cardsRemainingInDeck.clone();
			
			this.estimator = play.estimator;
		}

		/**
		 * @see java.util.concurrent.Callable#call()
		 */
		@Override
		public long[][] call() {
			long[] monteCarloCounts = new long[numCardsRemaining];
			long[] monteCarloSums = new long[numCardsRemaining];
			long[][] toRet = { monteCarloCounts, monteCarloSums };
			try {

				doMonteCarloSims(monteCarloCounts, monteCarloSums);
				
			} catch (Exception ex) {
				// WISH: Logging.
				System.err.println(ex);
				if (ASSERTIONS_ENABLED) {
					Thread t = Thread.currentThread();
					t.getUncaughtExceptionHandler().uncaughtException(t, ex);
					assert(false);
					return null;
				}
			}
			return toRet;
		}

		private void doMonteCarloSims(long[] monteCarloCounts, long[] monteCarloSums) {
			// Basic idea: shuffle the deck. Then play a game with the card
			// played in each position.
			// Then repeat.
			
			// Ensure that there are no weird do/undo errors...
			Card[][] boardClone = null;
			// ASSERTIONS_ENABLED is a final small primitive, so this is fine.
			if (ASSERTIONS_ENABLED) {
				boardClone = board.clone();
				for (int ii = 0; ii < 5; ii++)
					boardClone[ii] = board[ii].clone();
			}
			
			while (true) {
				Collections.shuffle(Arrays.asList(cardsRemainingInDeck).subList(0, 51 - numCardsPlayed));
				
				for (int i = 0; i < numCardsRemaining; i++) {
					doBoardPlayAndFreePos(freePositions, board, card, i, numCardsRemaining - 1);
					int value = getValueOfGamePlayedToEnd(estimator, board, cardsRemainingInDeck, freePositions,
							numCardsRemaining - 1, numCardsPlayed + 1); // Play a game
					monteCarloSums[i] += value;
					monteCarloCounts[i] += 1;
					undoBoardPlayAndFreePos(card, freePositions, board, i, numCardsRemaining - 1);
					assert (Arrays.deepEquals(board, boardClone));
					if (System.currentTimeMillis() >= endTime)
						return;
				}
			}
		}

	}

}
