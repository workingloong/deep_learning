package com.deeplearning.net;
import com.deeplearning.matrix.*;

public class Sigmoid implements ActivationFunction{
	public double fn(double z) {
		return 1.0/(1 + Math.exp(-1.0 * z));
	}
	
	public Array fn(Array z) {
		double[] fout = new double[z.size()];
		for(int i = 0; i < z.size(); i++) {
			fout[i] = fn(z.getElement(i));
		}
		return new Array(fout);
	}
	
	public double derivative(double z) {
		double a = fn(z);
		return a * (1.0 -a);
	}
	
	public Array derivative(Array z) {
		double[] d = new dsouble[z.size()];
		for(int i = 0; i < z.size(); i++) {
			double tempZ = z.getElement(i);
			d[i] = derivative(tempZ);
		}
		return new Array(d);
	}

}
