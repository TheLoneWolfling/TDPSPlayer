import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class TDPlayer implements PokerSquaresPlayer {

	private Card[][] boardState;
	BPANNE estimator = new BPANNE(25 * (1 + 1 + 1) + /*pointTable.length + */1, 20, 10, 1);
	private int numPlayed = 0;
	private PokerSquaresPointSystem scorer;
	private int divBy;
	public double learnRate = 0.1;
	int depth = 0;
	private Card[] allCards;

	private int normalizeMinMax(int[] pointTable) {
		// Scales points so they are within [-1,1], inclusive
		int min = pointTable[0];
		int max = pointTable[0];
		for (int entry : pointTable) {
			min = Math.min(min, entry);
			max = Math.max(max, entry);
		}
		return Math.max(Math.abs(min), Math.abs(max)) * 10;
	}


	@Override
	public void setPointSystem(PokerSquaresPointSystem system, long millis) {
		int[] pointTable = system.getScoreTable();
		this.divBy = normalizeMinMax(pointTable);
		this.scorer = system;
//		System.out.println(Arrays.toString(this.pointTable));
	}


	@Override
	public void init() {
		this.boardState = new Card[5][5];
		this.numPlayed = 0;
		this.allCards = new Card[52];
		for (int i = 0; i < 52; i++)
			allCards[i] = Card.getCard(i);
	}
	
	public double estimate() {
		//double[] pointTable = scorer.getScoreTable();
		double[] input = new double[25 * (1 + 1 + 1) + /*pointTable.length + */1];
		int ind = 0;
		for (int i = 0; i < 25; i++) {
			Card c = boardState[i / 5][i % 5];
			input[ind++] = c == null ? 0 : 1;
//			for (int s = 0; s < 4; s++)
//				input[ind++] = c == null ? 0: (c.getSuit() == s ? 1 : 0);
//			for (int s = 0; s < 13; s++)
//				input[ind++] = c == null ? 0 : (c.getRank() == s ? 1 : 0);
			input[ind++] = c == null ? 0 : c.getSuit() / 4.0;
			input[ind++] = c == null ? 0 : c.getRank() / 13.0;
		}
		//for (int i = 0; i < pointTable.length; i++)
		//	input[ind++] = pointTable[i] / (double) divBy;
		input[ind++] = scorer.getScore(boardState) / (double) divBy;
		if (ind != input.length) {
			System.out.println(Arrays.toString(input));
			throw new RuntimeException(ind + " " + input.length);
		}
//		double sumSq = 0;
//		for (int i = 0; i < input.length; i++) {
//			sumSq += Math.pow(input[i], 2);
//		}
//		double len = Math.sqrt(sumSq);
//
//		for (int i = 0; i < input.length; i++) {
//			input[i] /= len;
//		}
		
		return estimator.doEstimate(input, false);
		
	}

	@Override
	public int[] getPlay(Card card, long millisRemaining) {
		assert this.numPlayed < 25;
		int[] bestPlay = {-1, -1};
		for (int i = 0; i < 52; i++)
			if (allCards[i].equals(card)) {
				allCards[i] = allCards[51 - numPlayed];
				allCards[51 - numPlayed] = card;
			}
		
		double curEstMeanScore = getBestPlay(card, allCards,  51 - numPlayed, 25 - numPlayed, bestPlay, depth);
		
		double oldest = estimate();
		this.boardState[bestPlay[0]][bestPlay[1]] = card;
		double target;
		if (++this.numPlayed == 25)
			target = this.scorer.getScore(boardState) / (double) divBy;
		else
			target = curEstMeanScore;
		//if (this.numPlayed == 25)
		//	System.out.println("Err:\t" + (oldEst - target) + "\tTarget:\t" + target + "\toldest:\t" + oldEst + "\tcurEstMean:\t" + curEstMeanScore);
		this.estimator.update(target, learnRate);
//		System.out.print(numPlayed + " " + bestPlay[0] + " " + bestPlay[1] + ": " + curEstMeanScore + " " + numTies + " -> \n");
		return bestPlay;
	}
	
	public double getBestPlay(Card card, Card[] cardsLeft, int numCardsLeft, int numPlaysLeft, int[] bestPlay, int depth) {
		if (numPlaysLeft == 0) {
			return scorer.getScore(boardState);
		}
		boolean doRand = Math.random() > 0.99;
		double curEstMeanScore = Double.NEGATIVE_INFINITY;
		int numTies = 0;
		int[] nextBest = new int[] {-1, -1};
		for (int x = 0; x < 5; x++) {
			for (int y = 0; y < 5; y++) {
				if (this.boardState[x][y] != null)
					continue;
				this.boardState[x][y] = card;
				double estimatedValue;
				if (depth > 0) {
					double sum = 0;
					int num = 0;
					for (int i = 0; i < numCardsLeft; i++) {
						Card nc = cardsLeft[i];
						Card temp = cardsLeft[numCardsLeft - 1];
						cardsLeft[i] = temp;
						num += 1;
						sum += getBestPlay(nc, cardsLeft, numCardsLeft - 1, numPlaysLeft - 1, nextBest, depth - 1);
						cardsLeft[i] = nc;
						cardsLeft[numCardsLeft - 1] = temp;
					}
					estimatedValue = sum / num;
				}  else {
					estimatedValue = estimate();
					
				}
				if (doRand) {
					if (Math.random() * (++numTies) < 1) {
						bestPlay[0] = x;
						bestPlay[1] = y;
						curEstMeanScore = estimatedValue;
					}
				} else if (estimatedValue > curEstMeanScore || (estimatedValue == curEstMeanScore && Math.random() * (++numTies) < 1)) {
					bestPlay[0] = x;
					bestPlay[1] = y;
					curEstMeanScore = estimatedValue;
					if (estimatedValue > curEstMeanScore)
						numTies = 0;
				}
				if (depth > 1)
					System.out.println(x + " " + y + " " + estimatedValue + " " + depth);
				this.boardState[x][y] = null;
			}
		}
		return curEstMeanScore;
	}

	@Override
	public String getName() {
		return "TDPlayer";
	}
	
	@Override
	public String toString() {
		return this.estimator.toString();
	}

}
