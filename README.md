# This is a fork of Wi-Mind
This fork takes a look at using autoencoders to improve the original author's neural network approach for classification of the previously acquired data.  
It will only add some files into ["./FeatureExtractionModule/src/"]("/FeatureExtractionModule/src/") and update two files there to python 3.7.  
The rest of this repository was made by the original author.  

# Wi-Mind

The goal of this project was to infer with humans cognitive load. Its preliminary results were presented in UbitTention workshop article and a more detailed description of overall system was presented in master thesis at University of Ljubljana.

The project contains:
* wireless monitoring module (GNU Radio's *gr-radar* module) 
* feature extraction module (to filter and extract human breathing and heartrate features), along with neural network evaluation approach
* machine learning (Orange's workflows)
* band heart rate app (application for Android to acquire heart rate from Microsoft Band 2)
* and folder *study data*, which consists of the raw data from wireless module and from the cognitive load measuring app (not included in this project) 