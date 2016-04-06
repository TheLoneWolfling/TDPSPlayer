import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;


public class FinalTweaker {

	public static void main(String[] args) throws ClassNotFoundException, IOException {
		long seed = 3;
		Random r = new Random(seed);
		
		TDPlayer player = new TDPlayer();
		String curBestFile = "Player_0.1_258000_73.962.dat";
		//player.estimator = loadFromFile(curBestFile);
		player.learnRate = 0.1;
		//player.learnRate = 2.1544346900318823E-4;
		double maxSoFar = Double.NEGATIVE_INFINITY;
		int numTrials = 10000;
		PokerSquaresPointSystem sys = PokerSquaresPointSystem.getAmericanPointSystem();
		String curFile = "start.dat";
		saveToFile(player.estimator, curFile);
		for (long trial = 0; true ;trial += numTrials) {
			System.out.printf("%16d %.17f", trial, player.learnRate);
			double lR = player.learnRate;
			
			double sameMean = runNGames(r, sys, player, numTrials);
			BPANNE sameEst = player.estimator;
			

			System.out.printf(" %.16f", sameMean);
			
			player.estimator = loadFromFile(curFile);
			player.learnRate = lR * 2;
			double addMean = runNGames(r, sys, player, numTrials);
			BPANNE addEst = player.estimator;

			System.out.printf(" %.16f", addMean);

			player.estimator = loadFromFile(curFile);
			player.learnRate = lR / 10;
			double subMean = runNGames(r, sys, player, numTrials);
			BPANNE subEst = player.estimator;

			System.out.printf(" %.16f", subMean);
			
			double mean;
			if (addMean > sameMean && addMean > subMean) {
				player.learnRate = lR * 1.5;
				player.estimator = addEst;
				mean = addMean;
			} else if (subMean > addMean && subMean > sameMean) {
				player.learnRate = lR / 2;
				player.estimator = subEst;
				mean = subMean;
			} else {
				player.learnRate = lR;
				player.estimator = sameEst;
				mean = sameMean;
			}
			
			System.out.printf(" %.16f\n", mean);
			
			if (mean > maxSoFar) {
				maxSoFar = mean;
				saveToFile(player.estimator, "best_" + maxSoFar + "_" + player.learnRate + "_" + trial + ".dat");
			}
			curFile = mean + "_" + player.learnRate + "_" + trial + ".dat";
			saveToFile(player.estimator, curFile);
		}
	}
	private static double runNGames(Random r, PokerSquaresPointSystem sys, TDPlayer player, int numTrials) {
		long sum = 0;
		PokerSquares ps = new PokerSquares(player, sys);
		for (int j = 0; j < numTrials; j++) {
			ps.verbose = false;
			ps.setSeed(r.nextLong());
			sum += ps.play();
		}
		double mean = sum / (double) numTrials;
		return mean;
	}
	public static void saveToFile(Serializable s, String filename) {
		try
	      {
	         FileOutputStream fileOut = new FileOutputStream(filename);
	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
	         out.writeObject(s);
	         out.close();
	         fileOut.close();
	      }catch(IOException i)
	      {
	          i.printStackTrace();
	      }
	}
	public static BPANNE loadFromFile(String filename) throws IOException, ClassNotFoundException {
		FileInputStream fileIn  = null;
		ObjectInputStream in = null;
		try
	      {
			fileIn = new FileInputStream(filename);
			in = new ObjectInputStream(fileIn);
	        return (BPANNE) in.readObject();
	      } finally {
	    	  if (in != null)
	    		  in.close();
	    	  if (fileIn != null)
	    		  fileIn.close();
	      }
	}
		
		
}
