Implementation of Naive Bayes and Tree Augmented Network(TAN) for binary classification

The files are in ARFF format

For the TAN algorithm:

1) Prim's algorithm is used to find a maximum spanning tree. 
2) Initialize this process by choosing the first variable in the input file for Vnew
3) when there were ties in selecting maximum weight edges, used the following preference criteria: (1)  edges emanating from variables listed earlier in the input file, (2) if there were multiple maximum weight edges emanating from the first such variable, used bedges going to variables listed earlier in the input file.
To root the maximal weight spanning tree, pick the first variable in the input file as the root

The program is called bayes and accepts four command-line arguments as follows:
"bayes train-set-file test-set-file n|t"
where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN
