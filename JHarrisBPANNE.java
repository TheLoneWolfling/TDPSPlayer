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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;


// BPANNE: BackPropagation Artifical Neural Network Estimator

// The version included in the hand-in is "neutered" - it has had the learning infastructure
// removed to make it faster.

public class JHarrisBPANNE implements Serializable {
	
	private static final long serialVersionUID = 2L;
	final double[/*earlier layer*/][/*earlier layer node*/][/*later layer node*/] weights;
	
	final double[/*layer*/][/*node*/] cVals;
	
	final int[] topology;

 	private JHarrisBPANNE() { // Needed as otherwise this cannot be compiled without BPANNEOld2
		weights = null;
		cVals = null;
		topology = null;
	}
	
	
// 	public JHarrisBPANNE(BPANNEOld2 o) {
// 		this.topology = Arrays.copyOf(o.topology, o.topology.length);
// 		weights = new double[topology.length - 1][][];
// 		for (int i = 0; i < weights.length; i++) {
// 			weights[i] = new double[topology[i]][];
// 			for (int j = 0; j < weights[i].length; j++) {
// 				weights[i][j] = Arrays.copyOf(o.weights[i][j], topology[i + 1]);
// 			}
// 		}
// 		cVals = Arrays.copyOf(o.cVals, topology.length);
// 	}
	
	double doEstimate(double[] inputs) {
		assert (inputs.length == topology[0] - 1);
		final double[/*layer*/][/*node*/] cValsTemp = new double[cVals.length][];
		for (int i = 0; i < cVals.length; i++)
			cValsTemp[i] = cVals[i].clone();
		
		for (int i = 0; i < inputs.length; i++) {
			cValsTemp[0][i] = inputs[i];
		}
		
		cValsTemp[0][topology[0] - 1] = 1;
		
		for (int layer = 1; layer < topology.length; layer++) {
			for (int curNode = 0; curNode < topology[layer] - (layer == topology.length - 1 ? 0 : 1) /* bias*/; curNode++) {
				double sum = 0;
				for (int prevNode = 0; prevNode < topology[layer - 1]; prevNode++) {
					sum += weights[layer - 1][prevNode][curNode] * cValsTemp[layer - 1][prevNode];
				}
				cValsTemp[layer][curNode] = fwdFunc(sum);
			}
		}
		return cValsTemp[cValsTemp.length - 1][0];
	}

	private double fwdFunc(double in) {
		return Math.tanh(in);
	}

	@Override
	public String toString() {
		String toRet = "";
		toRet += "BPANNE[\n";
		for (int layer = 0; layer < topology.length-1; layer++) {
			for (int prevNode = 0; prevNode < topology[layer]; prevNode++) {
				for (int nextNode = 0; nextNode < topology[layer + 1]; nextNode++) {
					toRet += weights[layer][prevNode][nextNode] + " ";
				}
				toRet += "\n";
			}
			toRet += "\n";
		}
		toRet += "]\n";
		return toRet;
	}
	
	
	
// 	private static JHarrisBPANNE loadEstimatorFromFile(String filename) {
// 		// Blehh...
// 		
// 		FileInputStream fileIn = null;
// 		ObjectInputStream in = null;
// 		try {
// 			fileIn = new FileInputStream(filename);
// 			in = new ObjectInputStream(fileIn);
// 			BPANNEOld2 toRet = (BPANNEOld2) in.readObject();
// 			return new JHarrisBPANNE(toRet);
// 		} catch (IOException | ClassNotFoundException e) {
// 			e.printStackTrace();
// 			throw new RuntimeException("Cannot find estimator?");
// 		} finally {
// 			try {
// 				if (in != null)
// 					in.close();
// 			} catch (IOException e) {
// 				e.printStackTrace();
// 			}
// 			try {
// 				if (fileIn != null)
// 					fileIn.close();
// 			} catch (IOException e) {
// 				e.printStackTrace();
// 			}
// 		}
// 	}
	
// 	public static void saveToFile(Serializable s, String filename) {
// 		try
// 	      {
// 	         FileOutputStream fileOut = new FileOutputStream(filename);
// 	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
// 	         out.writeObject(s);
// 	         out.close();
// 	         fileOut.close();
// 	      }catch(IOException i)
// 	      {
// 	          i.printStackTrace();
// 	      }
// 	}

// 	public static void main(String[] args) {
// 		saveToFile(loadEstimatorFromFile("JHarrisTDEstimator2.dat"), "out.dat");
// 	}
}
