package com.deeplearning.net;
import com.deeplearning.matrix.*;

public interface ActivationFunction {
	
	public double fn(double z);
	
	public Array fn(Array z);
	
	public double derivative(double z);
	
	public Array derivative(Array z);
}
