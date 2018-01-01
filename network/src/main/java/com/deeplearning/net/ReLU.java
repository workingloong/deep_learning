package com.deeplearning.net;

import com.deeplearning.matrix.Array;

public class ReLU implements ActivationFunction{
	public double fn(double z) {
		return z > 0 ? z : 0;
	}
	
	public Array fn(Array z) {
		double[] fout = new double[z.size()];
		for(int i = 0; i < z.size(); i++) {
			fout[i] = fn(z.getElement(i));
		}
		return new Array(fout);
	}
	
	public double derivative(double z) {
		if(z > 0) {
			return 1.0;
		}
		return 0.0;
	}
	
	public Array derivative(Array z) {
		double[] d = new double[z.size()];
		for(int i = 0; i < z.size(); i++) {
			double tempZ = z.getElement(i);
			d[i] = derivative(tempZ);
		}
		return new Array(d);
	}

}
