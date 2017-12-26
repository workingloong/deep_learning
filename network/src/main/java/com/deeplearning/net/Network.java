package com.deeplearning.net;

import com.deeplearning.loss.*;
import com.deeplearning.matrix.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Network {
	private int layerNum;
	private Array sizes;
	private CostOperator cost;
	private List<Matrix> weights;
	private List<Array> biases;
	private ActivationFunction activate;
	
	public Network() {}
	
	public Network(Array sizes, CostOperator cost, ActivationFunction activate) {
		this.layerNum = sizes.size();
		this.sizes = sizes;
		this.cost = cost;
		this.activate = activate;
		weights = new ArrayList<Matrix>();
		biases = new ArrayList<Array>();
		weightInitialize(false);
	}
	
	public void weightInitialize(boolean large) {
		Random rand = new Random();
		for(int i = 1; i < sizes.size(); i++) {
			int neuronNum = (int) sizes.getElement(i);
			double[] nums = new double[neuronNum];
			for(int k = 0; k < nums.length; k++) {
				nums[k] = rand.nextGaussian();
			}
			biases.add(new Array(nums));
		}
		for(int i = 0; i < sizes.size() - 1; i++) {
			int m = (int) sizes.getElement(i + 1);
			int n = (int) sizes.getElement(i);
			double[][] w = new double[m][n];
			for(int j = 0; j < m; j++) {
				for(int k = 0; k < n; k++) {
					if(large) {
						w[i][j] = rand.nextGaussian();
					}else {
						w[i][j] = rand.nextGaussian()/Math.sqrt(n);
					}
				}
			}
			weights.add(new Matrix(w));
		}
	}
	public Array feedForward(Array a) {
		for(int i = 0; i < weights.size(); i++){
			Matrix w =  weights.get(i);
			Array b = biases.get(i);
			double[] wx = new double[w.getArrays().length];
			int k = 0;
			for(Array wi : w.getArrays()) {
				wx[k++] = wi.multiply(a);
			}
			a = activate.fn(b.add(new Array(wx)));
		}
		return a;
	}
	
	public void SGD(List<SampleData> trainData, int epoches, int miniBatchSize, double eta, double lambda) {
		int samplesNum = trainData.size();
		for(int i = 0; i < epoches; i++) {
			System.out.println("epoch : " + i);
			Collections.shuffle(trainData);
			int batches = trainData.size() / miniBatchSize;
			for(int j = 0; j < batches; j++) {
				List<SampleData> batchData = new ArrayList<SampleData>();
				for(int k = 0; k < miniBatchSize; k++) {
					batchData.add(trainData.get(j * miniBatchSize + k));
				}
				updateMiniBatch(batchData, eta, lambda, samplesNum);
			}
		}
	}
	
	public void updateMiniBatch(List<SampleData>  data, double eta, double lambda, int samplesNum) {
		Array[] nablaB = new Array[biases.size()];
		Matrix[] nablaW = new Matrix[weights.size()];
		for(int i = 0; i < data.size(); i++) {
			Array x = data.get(i).getFeatures();
			Array y = data.get(i).getLabels();
			backProp(x, y, nablaW, nablaB);
		}
		for(int i = 0; i < weights.size(); i++) {
			weights.get(i).multiplyConstant(1 - eta * (lambda) / samplesNum);
			nablaW[i].multiplyConstant(eta / data.size());
			Matrix tempM = weights.get(i).minus(nablaW[i]);
			weights.remove(i);
			weights.add(i, tempM);
			nablaB[i].multiplyConstant(eta / data.size());
			Array tempB = biases.get(i).minus(nablaB[i]);
			biases.remove(i);
			biases.add(i, tempB);
			
		}
		
		//self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		//self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		
		
	}
	
	public void backProp(Array x, Array y, Matrix[] nablaW, Array[] nablaB) {
		List<Array> deltaNablaB = new ArrayList<Array>();
		List<Matrix> deltaNablaW = new ArrayList<Matrix>();
		Array activation = x;
		List<Array> activations = new ArrayList<Array>();
		List<Array> zs = new ArrayList<Array>();
		activations.add(activation);
		for(int i = 0; i < weights.size(); i++) {
			Array z = weights.get(i).multiplyArray(activation).add(biases.get(i));
			zs.add(z);
			activation = activate.fn(z);
			activations.add(activation);
		}
		Array output = activations.get(activations.size() - 1);
		Array outZ = zs.get(zs.size()-1);
		Array deltaSigma = cost.derivative(output, y).dotMultiply(activate.derivative(outZ));
		deltaNablaB.add(deltaSigma);
		Matrix tempDeltaW = calDeltaW(deltaSigma, activations.get(activations.size() - 2));
		deltaNablaW.add(tempDeltaW);
		for(int i = weights.size() - 2; i >= 0; i--) {
			Array curZ = zs.get(i);
			Array prime = activate.derivative(curZ);
			deltaSigma = weights.get(i + 1).transposeMatrix().multiplyArray(deltaSigma).dotMultiply(prime);
			deltaNablaB.add(0, deltaSigma);
			tempDeltaW = calDeltaW(deltaSigma, activations.get(i));
			deltaNablaW.add(0,tempDeltaW);
		}
		for(int i = 0; i < nablaW.length; i++) {
			if(nablaW[i] == null) {
				int m = deltaNablaW.get(i).getRowNums();
				int n = deltaNablaW.get(i).getColNums();
				Array[] initialW = new Array[m];
				for(int k = 0; k < m; k++) {
					initialW[k] = new Array(new double[n]);
				}			
				nablaW[i] = new Matrix(initialW);
			}
			if(nablaB[i] == null) {
				double[] initialB = new double[deltaNablaB.get(i).size()];
				nablaB[i] = new Array(initialB);
			}
			nablaW[i] = nablaW[i].add(deltaNablaW.get(i));
			nablaB[i] = nablaB[i].add(deltaNablaB.get(i));
		}
	}
	
	public Matrix calDeltaW(Array sigma, Array a) {
		Matrix sigmaM = new Matrix(new Array[] {sigma});
		Matrix aM = new Matrix(new Array[] {a});
		return sigmaM.transposeMatrix().multiply(aM);
	}
	
	public boolean evaluate(Array inputData, Array label) {
		Array prob = feedForward(inputData);
		int preLabel = prob.getMaxIndex();
		boolean flag = false;
		if(label.getElement(preLabel) == 1) {
			flag = true;
		}
		return flag;
	}
}
