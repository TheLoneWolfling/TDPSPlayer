
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class BPANNE implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4199577866272821619L;
	final double[/*earlier layer*/][/*earlier layer node*/][/*later layer node*/] weights;
	final double[/*earlier layer*/][/*earlier layer node*/][/*later layer node*/] oldWeights;
	final double[/*layer*/][/*node*/] errs;
	final double[/*layer*/][/*node*/] cVals;
	final boolean[][] toDrop;
	double dropRate = 0.0;
	double momentum = 0.0;//0.9;
	double weightRange = 0.5;
	double weightBias = 0.0;
	final int[] topology;
	
	public BPANNE(int... topology) {
		this.topology = topology;
		for (int i = 0; i < topology.length - 1; i++)
			topology[i] += 1; // Bias nodes;
		Random r = new Random();
		weights = new double[topology.length - 1][][];
		oldWeights = new double[topology.length - 1][][];
		toDrop = new boolean[topology.length][];
		for (int i = 0; i < toDrop.length; i++) {
			toDrop[i] = new boolean[topology[i]];
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] = new double[topology[i]][];
			oldWeights[i] = new double[topology[i]][];
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = new double[topology[i + 1]];
				oldWeights[i][j] = new double[topology[i + 1]];
			}
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] = r.nextDouble() * weightRange * 2 - weightRange + weightBias;
				}
			}
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					weights[i][j][k] /= Math.sqrt(weights[i].length);
					oldWeights[i][j][k] = weights[i][j][k];
				}
			}
		}
		errs = new double[topology.length][];
		for (int i = 1; i < errs.length; i++) {
			errs[i] = new double[topology[i]];
		}
		
		cVals = new double[topology.length][];
		for (int i = 0; i < cVals.length; i++) {
			cVals[i] = new double[topology[i]];
			cVals[i][topology[i] - 1] = 1; // Bias
		}
	}
	
	public BPANNE(BPANNE o) {
		this.dropRate = o.dropRate;
		this.momentum = o.momentum;
		this.weightRange = o.weightRange;
		this.weightBias = o.weightBias;
		this.topology = Arrays.copyOf(o.topology, o.topology.length);
		weights = new double[topology.length - 1][][];
		oldWeights = new double[topology.length - 1][][];
		toDrop = new boolean[topology.length][];
		for (int i = 0; i < toDrop.length; i++) {
			toDrop[i] = Arrays.copyOf(o.toDrop[i], topology[i]);
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] = new double[topology[i]][];
			oldWeights[i] = new double[topology[i]][];
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] = Arrays.copyOf(o.weights[i][j], topology[i + 1]);
				oldWeights[i][j] = Arrays.copyOf(o.oldWeights[i][j], topology[i + 1]);
			}
		}
		errs = Arrays.copyOf(o.errs, topology.length);
		cVals = Arrays.copyOf(o.cVals, topology.length);
	}
	
	double doEstimate(double[] inputs, boolean isFinal) {
		for (int layer = 1; layer < topology.length - 1; layer++) {
			for (int curNode = 0; curNode < topology[layer]; curNode++) {
				toDrop[layer][curNode] = Math.random() < dropRate;
			}
		}
		if (inputs.length != topology[0] - 1)
			throw new RuntimeException(inputs.length + " " + (topology[0] - 1));
		for (int i = 0; i < inputs.length; i++) {
			cVals[0][i] = inputs[i];
		}
		cVals[0][topology[0] - 1] = 1;
		for (int layer = 1; layer < topology.length; layer++) {
			for (int curNode = 0; curNode < topology[layer] - (layer == topology.length - 1 ? 0 : 1) /* bias*/; curNode++) {
				if (!isFinal && toDrop[layer][curNode])
					continue;
				double sum = 0;
				for (int prevNode = 0; prevNode < topology[layer - 1]; prevNode++) {
					if (!isFinal && toDrop[layer - 1][prevNode])
						continue;
					sum += weights[layer - 1][prevNode][curNode] * cVals[layer - 1][prevNode];
				}
				if (isFinal)
					sum /= (1 - dropRate);
				cVals[layer][curNode] = fwdFunc(sum);
			}
		}
		return cVals[cVals.length - 1][0];
	}

	private double fwdFunc(double in) {
		return Math.tanh(in);
	}
	
	private double dervFunc(double out) {
		assert Math.abs(out) <= 1;
		return 1 - Math.pow(out,  2);
	}

	public void update(double estimate, double learnRate) {
		if (learnRate == 0)
			return;
		double err = estimate - cVals[cVals.length - 1][0];
		errs[errs.length - 1][0] = err * dervFunc(cVals[cVals.length - 1][0]);		
		for (int layer = topology.length - 2; layer >= 1; layer--) {
			for (int prevNode = 0; prevNode < topology[layer]; prevNode++) {
				if (toDrop[layer][prevNode])
					continue;
				errs[layer][prevNode] = 0;
				for (int nextNode = 0; nextNode < topology[layer + 1]; nextNode++) {
					if (toDrop[layer + 1][nextNode])
						continue;
					double nErr = errs[layer + 1][nextNode];
					double nWeight = weights[layer][prevNode][nextNode];
					errs[layer][prevNode] += nErr*nWeight;
				}
				errs[layer][prevNode] *= dervFunc(cVals[layer][prevNode]);
			}
		}
		for (int layer = topology.length - 2; layer >= 0; layer--) {
			for (int prevNode = 0; prevNode < topology[layer]; prevNode++) {
				if (toDrop[layer][prevNode])
					continue;
				for (int nextNode = 0; nextNode < topology[layer + 1]; nextNode++) {
					if (toDrop[layer + 1][nextNode])
						continue;
					double lErr = errs[layer + 1][nextNode];
					double cVal = cVals[layer][prevNode];
					double oldWeight = oldWeights[layer][prevNode][nextNode];
					double delta = learnRate * lErr * cVal + momentum * oldWeight;
					weights[layer][prevNode][nextNode] += delta;
					oldWeights[layer][prevNode][nextNode] = delta;
				}
			}
		}
	}
	
	public static void main(String[] args) {
		double[][][] toFit = {{{-1, -1}, {0.5}}, {{1, 1}, {0.5}}, {{-1, 1}, {-0.5}}, {{1, -1}, {-0.5}}};
		BPANNE b = new BPANNE(2, 10, 10, 1);
		System.out.printf("Score Mean\n");
		for (int i = 0; i < 200000; i++) {
			int numTrials = 1;
			double[] scores = new double[numTrials * toFit.length];
			double sum = 0;
			double min = Double.MAX_VALUE;
			double max = Double.MIN_VALUE;
			for (int n = 0; n < numTrials; n++) {
				for (int j = 0; j < toFit.length; j++) {
					double good = toFit[j][1][0];
					double act = b.doEstimate(toFit[j][0], false);
					b.update(good, 0.01);
					double score = Math.abs(good - act);
					scores[n*toFit.length +j] = score;
					sum += score;
					min = Math.min(score, min);
					max = Math.max(score, max);
				}
			}
			double mean = sum / (double) (numTrials * toFit.length);
			double sumSquDiffs = 0;
			for (int k = 0; k < numTrials; k++) {
				double diff = scores[k] - mean;
				sumSquDiffs += diff * diff;
			}
			double stdev = Math.sqrt(sumSquDiffs / (numTrials * toFit.length));
			//System.out.printf("Score Mean: %f, Standard Deviation: %f, Minimum: %f, Maximum: %f\n", mean, stdev, min, max);
			System.out.printf("%f\n", mean);
		}
//		for (int lay = 0; lay < b.topology.length - 1; lay++) {
//			for (int nodA = 0; nodA < b.topology[lay]; nodA++) {
//				for (int nodB = 0; nodB < b.topology[lay+1]; nodB++) {
//					System.out.print(b.weights[lay][nodA][nodB] + " ");
//				}
//				System.out.println();
//			}
//			System.out.println();
//		}
		for (int j = 0; j < toFit.length; j++) {
			System.out.println(toFit[j][0][0] + " " + toFit[j][0][1] + " -> " + b.doEstimate(toFit[j][0], true));
		}
			
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

}
