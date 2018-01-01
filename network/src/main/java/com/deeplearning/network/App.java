package com.deeplearning.network;

import com.deeplearning.matrix.*;
import com.deeplearning.net.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import com.deeplearning.loss.*;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
    		double[] networkSize = new double[] {784, 30, 10};
    		Array sizes= new Array(networkSize);
    		CostOperator cost = new QuadraticCost();
    		ActivationFunction activate = new Sigmoid();
        Network network = new Network(sizes, cost, activate);
        String trainFile = "/Users/workingloong/workspace/neural_network/java_dl/data/trainingData.txt";
        File file = new File(trainFile);
        List<SampleData> trainData = readTxtFile(file);
        int epoches = 10;
        int miniBatchSize = 100;
        double eta = 0.5;
        double lambda = 0.2;
        network.SGD(trainData, epoches, miniBatchSize, eta, lambda);
        System.out.println("finish train");
        String validationFile = "/Users/workingloong/workspace/neural_network/java_dl/data/trainingData.txt";
        file = new File(validationFile);
        List<SampleData> validationData = readTxtFile(file);
        int totalNum = validationData.size();
        double trueNum = 0.0;
        for(SampleData data : validationData) {
        		if(network.evaluate(data.getFeatures(), data.getLabels())) {
        			trueNum++;
        		}
        }
        System.out.println(trueNum / totalNum);
        
    }
    public static List<SampleData> readTxtFile(File file) {
    		List<SampleData> trainingData = new ArrayList<SampleData>();
    		try {
    			BufferedReader br = new BufferedReader(new FileReader(file));
    			String s = null;
    			while((s = br.readLine()) != null) {
    				trainingData.add(stringToData(s));
    			}
    		}catch(Exception e) {
    			e.printStackTrace();
    		}
    		return trainingData;
    }
    
    public static SampleData stringToData(String dataString) {
    		String[] fl = dataString.split(";");
		String[] featureString = fl[0].split(",");
		String[] labelString = fl[1].split(",");
		double[] featureValue = new double[featureString.length];
		double[] labelValue = new double[labelString.length];
		for(int i = 0; i < featureValue.length; i++) {
			featureValue[i] = Double.parseDouble(featureString[i]);
		}
		for(int i = 0; i < labelValue.length; i++) {
			labelValue[i] = Double.parseDouble(labelString[i]);
		}
		return new SampleData(new Array(featureValue), new Array(labelValue));
    }
}
