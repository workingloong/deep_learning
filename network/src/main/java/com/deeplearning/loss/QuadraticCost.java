package com.deeplearning.loss;

import com.deeplearning.matrix.*;

public class QuadraticCost implements CostOperator{
	
	public double fn(Array a, Array y) {
		double sum = a.minus(y).power(2).sum();
		return sum;
	}
	
	public Array derivative(Array a, Array y) {
		return a.minus(y);
	}

}
