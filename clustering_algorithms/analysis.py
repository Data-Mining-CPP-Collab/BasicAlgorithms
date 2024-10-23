import clustering as cl


"""
To actually use clustering on your data sets, create a new python file analysis.py that contains the code for your analysis, 
and that import clustering . This way you will not accidentally break the other test cases as you do your analysis 
(of course, if you find a bug in your clustering algorithm in this step, you should fix it and rerun the other tests).


Once you have implemented K-Means and DBSCAN algorithm and tested it thoroughly, it is time to apply it to 
your chosen data set. First you will have to consider what you expect a clustering to mean, and which attributes 
may be useful to get it. Before you apply the clustering algorithm do not forget to normalize your attributes so 
that they are all within the same range! You can always try your prepared data with an arbitrary k to make sure 
everything is working properly. Note that if you have many categorical attributes that you can not easily convert 
to numbers, you might want to consider implementing k- medoids, as that can work with such attributes.


The next step is to determine how to evaluate clusters! Given what you expect your clusters to mean, 
come up with a way to verify that the clusters produced by your algorithm are “reasonable”. 
What would it mean for two elements to be in the same cluster? What if they are in different clusters? 
If you have any “strong” intuition whether any particular instances should be in the same cluster 
or not (e.g. “DotA 2” and “LoL” should be in the same cluster, but “Chess” should be in a different one, 
if you expect your clusters to have to do with game play/mechanics), they can make excellent test cases 
(but that should not be the only criterion you look at). A good evaluation criterion would be something 
you can evaluate programmatically, e.g. a number you calculate or a graph that shows the clusters.


To effectively assess the quality of your clusters, you can use both intrinsic and extrinsic 
evaluation measures that was taught in class.

· Intrinsic Evaluation Measures: These metrics assess the clustering results based on the data itself without reference to the ground truth. Common intrinsic measures include the silhouette score, which evaluates how similar an object is to its own cluster compared to other clusters, may be used.

· Extrinsic Evaluation Measures: In contrast, extrinsic measures require external information, such as ground truth labels, to evaluate the quality of clusters. Examples include adjusted Rand index (ARI) and normalized mutual information (NMI), which compare the clustering results to known classifications. These metrics can provide a clear indication of how well your clustering algorithm performed in relation to expected groupings.

If you are using extrinsic evaluation, the you may think about generating the ground truth by 
labeling the data objects manually if the ground truth is not available.




With a way to evaluate your clusters in hand, you are now ready to actually calculate clusters. 
To do so, you will need to find good values for the parameters. 
For the number of clusters you can start with an intuitive guess, 
or simply start with 1 and increase it one by one, and evaluate each result using your evaluation criterion until you are happy 
(it may happen that there is more than one “good” value; if this happens, be sure to mention it in your report). 
For the termination criteria, you may want to use both (with a relatively large n ), to prevent the algorithm from “oscillating”, 
but to stop once the clusters do not change anymore. It may be helpful to print the movement distance of 
the cluster centers in each iteration to get a better picture of what is happening.


Finally, once you have clusters that satisfy your criterion, you will need to think about how to present them. 
For this, consider what “defines” each cluster (this will likely not be something you can automate), 
and describe each cluster in these terms. Please do not include a complete list of instances for each cluster in your report 
(especially not if there are thousands such instances). If any individual cluster is small and interesting, 
you can list its element, though.


Note: It may happen that there is no “good” clustering. In this case, make sure to note which parameter 
settings you tried, and why none of them lead to a satisfactory outcome
"""