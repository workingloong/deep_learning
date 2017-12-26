package com.deeplearning.loss;

import com.deeplearning.matrix.Array;

public interface CostOperator {
	public double fn(Array a, Array y);
	
	public Array derivative(Array a, Array y);
}
