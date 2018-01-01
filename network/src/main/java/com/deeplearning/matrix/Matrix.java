package com.deeplearning.matrix;

public class Matrix extends Array{
	private Array[] arrays;
	
	public Matrix(double[][] x) {
		arrays = new Array[x.length];
		for(int i = 0; i < x.length; i++) {
			arrays[i] = new Array(x[i]);
		}
	}
	
	public Matrix(Array[] x) {
		this.arrays = x;
	}
	
	public Matrix add(Matrix x){
		check(arrays.length, x.arrays.length);
		Array[] z = new Array[arrays.length];
		for(int i = 0; i < arrays.length; i++) {
			z[i] = arrays[i].add(x.arrays[i]);
		}
		return new Matrix(z);
	}
	
	public Matrix minus(Matrix x){
		check(arrays.length, x.arrays.length);
		Array[] z = new Array[arrays.length];
		for(int i = 0; i < arrays.length; i++) {
			z[i] = arrays[i].minus(x.arrays[i]);
		}
		return new Matrix(z);
	}
	
	public Matrix dotMultiply(Matrix x){
		check(arrays.length, x.arrays.length);
		Array[] z = new Array[arrays.length];
		for(int i = 0; i < arrays.length; i++) {
			z[i] = arrays[i].dotMultiply(x.arrays[i]);
		}
		return new Matrix(z);
	}
	
	public Matrix multiply(Matrix x){
		check(arrays[0].size(), x.arrays.length);
		double[][] z = new double[arrays.length][x.arrays[0].size()];
		Matrix xt = x.transposeMatrix();
		for(int i = 0; i < arrays.length; i++) {
			for(int j = 0; j < xt.arrays.length; j++) {
				z[i][j] = arrays[i].multiply(xt.arrays[j]);
			}
		}
		return new Matrix(z);
	}
	
	public Array multiplyArray(Array x){
		check(arrays[0].size(), x.size());
		double[] z = new double[arrays.length];
		for(int i = 0; i < arrays.length; i++) {
			z[i] = arrays[i].multiply(x);
		}
		return new Array(z);
	}
	
	public Matrix multiplyConstant(double num) {
		double[][] z = new double[arrays.length][arrays[0].size()];
		for(int i = 0; i < arrays.length; i++) {
			for(int j = 0; j <arrays[0].size(); j++) {
				z[i][j] = arrays[i].getElement(j) * num;
			}
		}
		return new Matrix(z);
	}
	
	public Matrix transposeMatrix(){
		double[][] z = new double[arrays[0].size()][arrays.length];
		for(int i = 0; i < arrays.length; i++) {
			for(int j = 0; j < arrays[0].size(); j++) {
				z[j][i] = arrays[i].getElement(j);
			}
		}
		return new Matrix(z);
	}
	
	
	public void printMatrix() {
		for(Array array : arrays) {
			array.print();
		}
		System.out.println("");
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(Array array : arrays) {
			sb.append(array.toString());
			sb.append("\n");
		}
		return sb.toString();
	}
	
	public Array getArrayElement(int i) {
		return arrays[i];
	}
	
	public Array[] getArrays() {
		return arrays;
	}
	
	public int getRowNums() {
		return arrays.length;
	}
	
	public int getColNums() {
		if(arrays == null || arrays.length == 0) {
			return 0;
		}
		return arrays[0].size();
	}


}

