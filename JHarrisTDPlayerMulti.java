
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

// TODO: Auto-generated Javadoc
// XXX: This shouldn't work / this is a bandaid patch.
// BUG: Fix me please
// TODO: I should look at this when I have time, but not urgent
// WISH: If I had infinite time I'd do this, but I don't and my time is
//       probably better spent elsewhere. (Lower priority TODO, give or take)

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
	// TODO: figure out optimum when including hyperthreading
	private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

	private static final JHarrisBPANNE estimator = loadEstimatorFromFile("JHarrisTDEstimator.dat");

	private static final ExecutorService pool = Executors.newFixedThreadPool(NUM_THREADS);

	private static final CompletionService<long[][]> completionService = new ExecutorCompletionService<long[][]>(pool);

	public static final boolean ASSERTIONS_ENABLED = areAssertionsEnabled();

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

	// TODO: pull into JHarrisBPANNE
	private static final PokerSquaresPointSystem POINT_SYSTEM = PokerSquaresPointSystem.getAmericanPointSystem();

	// Should be 1000
	// WISH: change to offset and scale according to *actual* min/max scores
	// possible.
	// But there is no easy way to figure out either of those.
	private static final int ESTIMATOR_SCALE = normalizeMinMax(POINT_SYSTEM.getScoreTable());

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
	private static int normalizeMinMax(int[] pointTable) {
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

	/**
	 * Loads estimator from file.
	 * 
	 * Note: insecure, not that this should make a difference for this case
	 * 
	 * TODO: pull into JHarrisBPANNE TODO: add override for
	 * loadEstimatorFromFile(File file)
	 *
	 * @param filename
	 *            filename to load estimator from
	 * @return the JHarrisBPANNE that was previously written to said file
	 */
	static JHarrisBPANNE loadEstimatorFromFile(String filename) {
		// Blehh...
		// My kingdom for a with statement!

		FileInputStream fileIn = null;
		ObjectInputStream in = null;
		try {
			fileIn = new FileInputStream(filename);
			in = new ObjectInputStream(fileIn);
			return (JHarrisBPANNE) in.readObject();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
			throw new RuntimeException("Cannot find estimator?");
			// TODO: replace with rethrowing exception and propagate through
			// Question: what type of exception should be thrown?
			// UnableToLoadEstimator(String filename)?
		} finally {
			try {
				if (in != null)
					in.close();
			} catch (IOException e) {
				e.printStackTrace();
				// WISH: add logging here
				// Here is where I'd log something, if I had a logger.
			}
			try {
				if (fileIn != null)
					fileIn.close();
			} catch (IOException e) {
				e.printStackTrace();
				// WISH: add logging here
				// Here is where I'd log something, if I had a logger.
			}
		}
	}

	/**
	 * Instantiates a new JHarrisTDPlayerMulti
	 * 
	 * Only used for debugging, really.
	 */
	public JHarrisTDPlayerMulti() {
		// WISH: log instead?
		System.out.println(getName() + " using " + NUM_THREADS + " threads.");
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
		assert (Arrays.equals(system.getScoreTable(), POINT_SYSTEM.getScoreTable()));

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

		// TODO: run a training run with it set to only use "useful" positions
		// i.e. only one position per unique set of {row, column}. (That's a
		// set, so {row, column} == {column, row}.)
		// Should make it faster / better.
		// "Should".

		if (numCardsRemaining > 1) { // Last card played is trivial.

			// TODO: could probably move startTime's declaration inside this
			// block, which would
			// shave a (tiny) amount of time off of the first / last plays.

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
				estimatedValues[i] = getValueForBoardPos(board);
				undoBoardPlay(board, freePositions[i]);
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
					// TODO: replace with lerp function?
					value = estimatedValues[j] * ESTIMATION_WEIGHT
							+ monteCarloSums[j] / (double) monteCarloCounts[j] * (1 - ESTIMATION_WEIGHT);

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
	 * Check if assertions are enabled.
	 * 
	 * Used for expensive "setup" for assertions
	 * 
	 * TODO: Find a better way to check this.
	 *
	 * @return true, if assertions are enabled
	 */
	public static boolean areAssertionsEnabled() {
		boolean checkAssert = false;
		assert checkAssert = true; // Side effect is intentional!
		if (checkAssert)
			System.out.println("Assertions are enabled"); // WISH: Proper
															// logging
		return checkAssert;
	}

	/**
	 * Gets the estimated mean score for a board.
	 * 
	 * Warning: this function cannot change behavior without a full training
	 * run. (Optimizations or sanity checks are fine; modifications of the
	 * inputs used are not.)
	 * 
	 * TODO: move into JHarrisBPANNE
	 *
	 * @param Card[5][5]
	 *            board the board to estimate the score of
	 * @return the estimated mean score of the board
	 */
	private static double getValueForBoardPos(Card[/* 5 */][/* 5 */] board) {

		boardSanityCheck(board);

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
		int boardScore = POINT_SYSTEM.getScore(board);
		if (numFree == 0)
			return boardScore;
		input[ind++] = boardScore / (double) ESTIMATOR_SCALE;

		assert (ind == input.length);

		return estimator.doEstimate(input) * (double) ESTIMATOR_SCALE;
	}

	/**
	 * Undoes a board play.
	 *
	 * @param Card[5][5]
	 *            board the board to undo the play on
	 * @param int[2]
	 *            pos the position to undo the play on
	 */
	private static void undoBoardPlay(Card[/* 5 */][/* 5 */] board, int[/* 2 */] pos) {

		boardSanityCheck(board);

		assert (pos != null);

		assert (pos.length == 2);

		assert (pos[0] >= 0);
		assert (pos[0] < 5);

		assert (pos[1] >= 0);
		assert (pos[1] < 5);

		assert (board[pos[0]][pos[1]] != null);

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
	private static int getValueOfGamePlayedToEnd(Card[/* 5 */][/* 5 */] board, Card[/* 52 */] cardsRemainingInDeck,
			int[/* 25 */][/* x,y */] freePositions, int numCardsRemaining, int numCardsPlayed) {
		// Recursive

		doSanityCheck(board, cardsRemainingInDeck, freePositions, numCardsRemaining, numCardsPlayed);

		// Base case:
		if (numCardsRemaining == 0)
			return POINT_SYSTEM.getScore(board);

		// If assertions are enabled, make a copy of the board to check that the
		// do / undo setup is sound
		Card[][] temp = null;
		if (ASSERTIONS_ENABLED) {
			temp = board.clone();
			for (int ii = 0; ii < 5; ii++)
				temp[ii] = board[ii].clone();
		}

		// TODO: pull this into a function of its own, as it's duplicated with
		// the
		// main getPlay function (just after starting the Monte Carlo threads)
		int bestIndex = 0;
		double best = Double.NEGATIVE_INFINITY;
		Card card = cardsRemainingInDeck[numCardsPlayed]; // Grab next card...
		for (int j = 0; j < numCardsRemaining; j++) {
			doBoardPlay(board, card, freePositions[j]); // Try putting it in a
														// position...
			double value = getValueForBoardPos(board); // Get the estimated
														// value...
			undoBoardPlay(board, freePositions[j]); // Remove the card from said
													// position

			assert (Arrays.deepEquals(board, temp)); // Check that we haven't
														// messed anything up

			// Due to the estimator ties appear so rarely that it's not even
			// worth
			// accounting for them.
			if (value > best) { // If it's the best we've see so far...
				bestIndex = j; // record it.
				best = value;
			}
		}
		assert (Arrays.deepEquals(board, temp)); // Check (again) that we
													// haven't messed anything
													// up
		// Probably not needed.

		doBoardPlayAndFreePos(freePositions, board, card, bestIndex, numCardsRemaining - 1); // Do
		// the
		// best
		// play...

		int toRet = getValueOfGamePlayedToEnd(board, cardsRemainingInDeck, freePositions, numCardsRemaining - 1,
				numCardsPlayed + 1); // Recurse!
		undoBoardPlayAndFreePos(freePositions, board, bestIndex, numCardsRemaining - 1); // Undo
		// said
		// play...

		assert (Arrays.deepEquals(board, temp));// Check (yet again) that we
												// haven't messed anything up

		return toRet; // Return the value of the game played to the end
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
	private static void undoBoardPlayAndFreePos(int[][] freePositions, Card[][] board, int index, int numCardsLeft) {
		swap(freePositions, index, numCardsLeft); // Remove the card from the
													// board...
		undoBoardPlay(board, freePositions[index]); // And record the play
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
		int id = card.getCardId();
		// Might be slightly faster to just check reference equality,
		// but I don't know if it's guaranteed that there is exactly one card
		// with a given rank/suit floating around.

		// TODO: check if cardA == cardB is sound.
		for (int i = 0; i < cardsRemainingInDeck.length; i++)
			if (cardsRemainingInDeck[i].getCardId() == id)
				return i;
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
	private static void boardSanityCheck(Card[][] board) {
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

		boardSanityCheck(board);

		// TODO: pull out these static checks to a static assertion method

		assert (MILLIS_TIME_BUFFER >= 0);

		assert (MILLIS_TIME_FINAL >= 0);

		assert (ESTIMATION_WEIGHT >= 0);
		assert (ESTIMATION_WEIGHT <= 1);

		assert (NUM_THREADS >= 1);

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
	 */
	public static void main(String[] args) {
		long score = 0;
		for (long i = 1;; i++) { // optimistic!
			JHarrisTDPlayerMulti player = new JHarrisTDPlayerMulti();
			PokerSquares ps = new PokerSquares(player, POINT_SYSTEM);
			int playScore = ps.play();
			score += playScore;
			System.out.println(String.format("%16d%16d%16f", i, playScore, score / (double) i));
		}
	}

	private static class MonteCarloCallable implements Callable<long[][]> {
		// The below all have the same comments as JHarrisTDPlayerMulti
		private Card[][] board;
		private double endTime;
		private int numCardsRemaining;
		private int numCardsPlayed;
		private Card[] cardsRemainingInDeck;
		private int[][] freePositions;

		// The card that we have to play currently.
		private Card card;

		/**
		 * Instantiates a new MonteCarloCallable.
		 * 
		 * TODO: look at using an integer type instead of double for endTime
		 * TODO: Sanity checks?
		 *
		 * @param play
		 *            the parent JHArrisTDPlayer
		 * @param card
		 *            the card that was flipped up
		 * @param endTime
		 *            the time that we should return at
		 */
		public MonteCarloCallable(JHarrisTDPlayerMulti play, Card card, double endTime) {
			// So many defensive copies :/
			// WISH: look at having the main player instead keep NUM_THREADS
			// copies of everything?

			// TODO: look at using clone instead?

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

		/**
		 * @see java.util.concurrent.Callable#call()
		 */
		@Override
		public long[][] call() {
			try {
				// Ensure that there are no weird do/undo errors...
				Card[][] temp = null;
				// ASSERTIONS_ENABLED is static final, so this "should" be fine
				// across multiple threads?
				// TODO: ensure that this access is legal.
				if (ASSERTIONS_ENABLED) {
					temp = board.clone();
					for (int ii = 0; ii < 5; ii++)
						temp[ii] = board[ii].clone();
				}

				long[] monteCarloCounts = new long[numCardsRemaining];
				long[] monteCarloSums = new long[numCardsRemaining];

				// Basic idea: shuffle the deck. Then play a game with the card
				// played in each position.
				// Then repeat.

				// The inner loop is implicit.
				// TODO: look at replacing this loop with while(True) { for (i
				// in range(numCardsRemaining) {if timeout break both loops}}

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
							numCardsRemaining - 1, numCardsPlayed + 1); // Play
																		// a
																		// game
					monteCarloSums[i] += value;
					monteCarloCounts[i] += 1;
					undoBoardPlayAndFreePos(freePositions, board, i, numCardsRemaining - 1);
					assert (Arrays.deepEquals(board, temp));

					i += 1;
				}
				return new long[][] { monteCarloCounts, monteCarloSums }; // So
																			// much
																			// garbage...
			} catch (Exception ex) {
				System.err.println(ex);
				if (ASSERTIONS_ENABLED) {
					Thread t = Thread.currentThread();
					t.getUncaughtExceptionHandler().uncaughtException(t, ex);
					return null;
				}
				return new long[][] { new long[numCardsRemaining], new long[numCardsRemaining] }; // i.e.
																									// just
																									// act
																									// as
																									// though
																									// we
																									// didn't
																									// do
																									// anything
																									// at
																									// all
			}
		}

	}

}
