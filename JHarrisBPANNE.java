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
import java.util.Arrays;

import jdk.nashorn.internal.runtime.JSONFunctions;


// BPANNE: BackPropagation Artifical Neural Network Estimator

// The version included here is "neutered" - it has had the learning infrastructure
// removed

public class JHarrisBPANNE implements Serializable {
	
	private static final long serialVersionUID = 2L;
	
	final double[/*earlier layer*/][/*earlier layer node*/][/*later layer node*/] weights;
	
	final int[/*# layers*/] topology; // # nodes in each layer - input layer == layer 0. Includes bias nodes.

 	private JHarrisBPANNE() { // Final variables and serializability don't mix well.
		weights = null;
		topology = null;
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
 										  // as there is no next layer for the bias node to connect to!
 			return topology[layer];
 		return topology[layer] - 1;
 	}
	
 	double doEstimate(double[] inputs) {
 		
 		// There are two ways to do bias values for neural network nodes
 		// Either have separate bias values
 		// Or have "nodes" that are never updated with a fixed activation of 1
 		// I take the second route, as it is easier when one is not using a matrix library.
 		
		assert (inputs.length == getLayerSizeNoBias(0));
		
		final double[/*layer*/][/*node*/] nodeActivations = new double[getNumLayers()][];
		
		// If we were learning we'd hang on to nodeActivations for the backprop phase.
		
		for (int i = 0; i < getNumLayers(); i++) {
			nodeActivations[i] = new double[getLayerSizeWithBias(i)];
			
			nodeActivations[i][getLayerSizeWithBias(i) - 1] = 1; // Bias nodes set to activation of 1
		}
		
		for (int i = 0; i < getLayerSizeNoBias(0); i++) { // Could use System.arraycopy here I suppose
			nodeActivations[0][i] = inputs[i];    // Frankly, I find this clearer.
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
				//Run it through the activation function...
				nodeActivations[nextLayer][nextNode] = fwdFunc(sum);
				//And assign it to the activation of said node.
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

}
