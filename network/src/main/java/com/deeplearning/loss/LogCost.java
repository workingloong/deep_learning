package com.deeplearning.loss;

import com.deeplearning.matrix.Array;

public class LogCost implements CostOperator{
	public double fn(Array a, Array y) {
		//loss function = -y*log(a)-(1-y)*log(1-a)
		double sum = 0;
		for(int i = 0; i < a.size(); i++) {
			if(a.getElement(i) == 0) {
				sum += 1*Math.pow(10, 10);
			}else {
				sum += -1 * y.getElement(i) * Math.log(a.getElement(i)) - 
						(1-y.getElement(i)) * Math.log(1-a.getElement(i));
			}
			
		}
		return sum;
	}
	
	public Array derivative(Array a, Array y) {
		double[] d = new double[a.size()];
		for(int i = 0; i < a.size(); i++) {
			double curA = a.getElement(i);
			double curY = y.getElement(i);
			d[i] = (curA - curY)/(curA * (1.0 - curA));
		}
		return new Array(d);
	}

}
