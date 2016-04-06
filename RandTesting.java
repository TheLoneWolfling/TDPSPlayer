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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;


public class RandTesting {

	public static void main(String[] args) {
		TDPlayer player = new TDPlayer();
		PokerSquaresPointSystem sys = PokerSquaresPointSystem.getAmericanPointSystem();
		run(player, sys);
	}
	public static void run(TDPlayer player, PokerSquaresPointSystem sys) {
//		System.out.println("Random Parameterized Poker Squares Testing:");
		
		long trial = 0;
		double maxMean = Double.NEGATIVE_INFINITY;
		for (int x = 0; x < 1000; x++) {
			int numTrials = 1000;
			int[] scores = new int[numTrials];
			long sum = 0;
			long min = Long.MAX_VALUE;
			long max = Long.MIN_VALUE;
			PokerSquares ps = new PokerSquares(player, sys);
			for (int j = 0; j < numTrials; j++) {
				trial += 1;
				ps.verbose = false;
				scores[j] = ps.play();
				sum += scores[j];
				min = Math.min(scores[j], min);
				max = Math.max(scores[j], max);
//				System.out.println(trial);
			}
			double mean = sum / (double) numTrials;
			double sumSquDiffs = 0;
			for (int i = 0; i < numTrials; i++) {
				double diff = scores[i] - mean;
				sumSquDiffs += diff * diff;
			}
			double stdev = Math.sqrt(sumSquDiffs / numTrials);
			double errIsh = stdev / Math.sqrt(numTrials);
			System.out.printf("%d\t%f\t%f\t%f\t%f\n", trial, mean, stdev, errIsh, errIsh / mean);
			if (mean > maxMean) {
				maxMean = mean;
				String filename = "Player_" + player.learnRate + "_" + trial + "_" + maxMean + ".dat";
				try
			      {
			         FileOutputStream fileOut = new FileOutputStream(filename);
			         ObjectOutputStream out = new ObjectOutputStream(fileOut);
			         out.writeObject(player.estimator);
			         out.close();
			         fileOut.close();
			      }catch(IOException i)
			      {
			          i.printStackTrace();
			      }
			}
		}
	}
}
