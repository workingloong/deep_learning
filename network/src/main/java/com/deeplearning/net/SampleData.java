package com.deeplearning.net;

import com.deeplearning.matrix.*;

public class SampleData {
	private Array features;
	private Array labels;
	
	public SampleData(Array features, Array labels) {
		this.features = features;
		this.labels = labels;
	}
	
	public Array getFeatures() {
		return features;
	}
	
	public Array getLabels() {
		return labels;
	}
}
