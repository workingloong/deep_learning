package com.deeplearning.matrix;

public class Array{
	private double[] nums;
	
	public Array() {
		this.nums = null;
	}
	
	public Array(double[] nums) {
		this.nums = nums;
	}
	
	public void check(int xLen, int yLen) {
		if(xLen != yLen ) {
			throw new RuntimeException("the length(x) does not equel to length(y); " + "xlen = " + xLen + ", yLen = " + yLen);
		}
	}
	
	public int size() {
		return nums.length;
	}
	
	public double getElement(int i) {
		return nums[i];
	}
	
	public Array add(Array x) {
		check(nums.length, x.nums.length);
		double[] z = new double[x.nums.length];
		for(int i = 0; i < x.nums.length; i++) {
			z[i] = nums[i] + x.nums[i];
		}
		return new Array(z);
	}
	
	public Array minus(Array x) {
		check(nums.length, x.nums.length);
		double[] z = new double[x.nums.length];
		for(int i = 0; i < x.nums.length; i++) {
			z[i] = nums[i] - x.nums[i];
		}
		return new Array(z);
	}
	
	public Array dotMultiply(Array x) {
		check(nums.length, x.nums.length);
		double[] z = new double[x.nums.length];
		for(int i = 0; i < x.nums.length; i++) {
			z[i] = nums[i] * x.nums[i];
		}
		return new Array(z);
	}
	
	public Array power(double n) {
		double[] z = new double[nums.length];
		for(int i = 0; i < nums.length; i++) {
			z[i] = Math.pow(nums[i], n);
		}
		return new Array(z);
	}
	
	public double multiply(Array x) {
		check(nums.length, x.nums.length);
		double z = 0.0;
		for(int i = 0; i < x.nums.length; i++) {
			z += nums[i] * x.nums[i];
		}
		return z;
	}
	
	public Array multiplyConstant(double num) {
		double[] z = new double[nums.length];
		for(int i = 0; i < nums.length; i++) {
			z[i] = nums[i] * num;
		}
		return new Array(z);
	}
	
	public void print() {
		for(double num : nums) {
			System.out.print(num + ", ");
		}
		System.out.println("");
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(double num : nums) {
			sb.append(num);
			sb.append(", ");
		}
		return sb.toString();
	}
	
	public double sum() {
		double sum = 0;
		for(int i = 0; i < nums.length; i++) {
			sum += nums[i];
		}
		return sum;
	}
	
	public int getMaxIndex() {
		double max = nums[0];
		int index = 0;
		for(int i = 1; i < nums.length; i++) {
			if(nums[i] > max) {
				max = nums[i];
				index = i;
			}
		}
		return index;
	}
	
}

