package com.deeplearning.network;

import com.deeplearning.matrix.*;
import java.util.*;

public class TestApp {
	public static void main(String[] args) {
		
		List<Matrix> listM = new ArrayList<Matrix>();
		listM.add(createMatrix(1.0));
		listM.get(0).multiplyConstant(4.0);
		System.out.println(listM.get(0));
		Matrix m = listM.get(0).minus(createMatrix(2.0));
		listM.remove(0);
		listM.add(0, m);
		System.out.println(listM.get(0));
	}
	
	public static Matrix createMatrix(double num) {
		Array[] arrays = new Array[3];
		for(int i = 0; i < arrays.length; i++) {
			double[] nums = new double[]{num,num,num};
			arrays[i] = new Array(nums);
		}
		return new Matrix(arrays);
	}

}
