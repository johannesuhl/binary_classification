# Visual-analytical tools to evaluate and compare the outputs of large numbers of binary classifiers

There are many applications which require the comparison and evaluation of large numbers of binary classification results, among each other and/or with respect to a reference classification. Such applications include e.g. the results of systematic hyperparameter tuning of machine-learning based binary classifiers, or different classifications of built-up versus not built-up areas derived from satellite imagery or other geospatial data sources (cf. Uhl et al. 2018, Fig. 5; and Uhl & Leyk 2019, Fig. 9).

## multi_classifier_binary_comparison.py
The script multi_classifier_binary_comparison.py provides a visual method to compare the binary classification outcomes between each other and with respect to the reference labels, by means of a heatmap indicating the positive (green) and negative (blue) reference labels (top row), and the corresponding labels of an arbitrary number of classifiers, sorted descendingly by a user-defined accuracy measure (e.g., the F-measure) from top to bottom. This allows to visually identify false positive and false negative instances and the commonalities and differences between different classifiers.

The script simulates multiple binary classification outcomes using a multilayer perceptron.

Classifying own simulated data:

<img width="750" alt="Classifying own simulated data" src="https://github.com/johannesuhl/binary_classification/blob/main/multiple_binary_classifier_comparisonOwn.jpg">

Classifying the Connectionist Bench (Sonar, Mines vs. Rocks) Data Set:

<img width="750" alt="Classifying the Connectionist Bench (Sonar, Mines vs. Rocks) Data Set" src="https://github.com/johannesuhl/binary_classification/blob/main/multiple_binary_classifier_comparisonSonar.jpg">

Classifying the  Pima Indians Diabetes Database:

<img width="750" alt="Classifying the  Pima Indians Diabetes Database" src="https://github.com/johannesuhl/binary_classification/blob/main/multiple_binary_classifier_comparisonDiabetes.jpg">

## References:

Uhl, J. H., Zoraghein, H., Leyk, S., Balk, D., Corbane, C., Syrris, V., & Florczyk, A. J. (2018). Exposing the urban continuum: Implications and cross-comparison from an interdisciplinary perspective. International Journal of Digital Earth. https://www.tandfonline.com/doi/10.1080/17538947.2018.1550120

Uhl, J. H., & Leyk, S. (2020). Towards a novel backdating strategy for creating built-up land time series data using contemporary spatial constraints. Remote Sensing of Environment, 238, 111197. https://www.sciencedirect.com/science/article/pii/S0034425719302093

## Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.

https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 

## Pima Indians Diabetes Database
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
