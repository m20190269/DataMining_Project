# DataMining_Project
Data is regarding a fictional
insurance company in Portugal regarding 10.290 Customers. 
As a Data Mining/Analytic Consultant, we were asked develop a Customer Segmentation in such
a way that would be possible for the Marketing Department to better understand all the
different Customers’ Profiles. 


To accomplish the goals of this project the work needed to be separated into six parts:
1.	Cleaning the Dataset (dealing with missing values and imputation and Inconsistency errors)
2.	Transforming the Variables (creating attributes/features as well as doing some dimensionality reduction)
3.	Detecting Outliers (detecting extreme values to further remove or not some outliers)
4.	Cluster Analysis
5.	Building the Final Model
6.	Reintroducing the Outliers
7.	Clusters Description

First, we separated the data into Client (the profiling attributes) and Consumption (the segmentation attributes)
as mentioned before and, in Client, we also separated the variables into numerical and categorical variables. 
We will perform Cluster Analysis individually in those 3 groups. 
Before the analysis we also normalized all the numerical variables to change them to a common scale without distorting
differences in the ranges of values.
For this Analysis we tried different clustering strategies, as like:
•	Partitioning Methods (K-means)
•	Hierarchical Methods (Agglomerative)
•	Combined K-means with Agglomerative
•	K-modes
•	Self-Organizing Maps
•	Mean-Shift Clustering
•	Density-based Clustering


For the end model choice we took in account (for the numerical variables) the Euclidian distances from each cluster
member to his centroid, and choosed the group of clusters that minimized those distances.

For the outlier reintroduction we used a classification tree to classify these elements as one of our seven clusters obtained:

1.	Customers that spend a significant amount of their salary in premiums with high customer monetary value. Most of
the premiums go for Household (≈44%) followed by Motor (≈22%) and Health (≈18%).
2.	Customers with a medium customer monetary value where most of their premiums go for Motor (≈35%) and Health (≈34%),
followed by Household (≈18%). Here we see a main differentiator on the two customers’ profiles here present:
  a)	One with high-fidelity time (≈36 years);
  b)	Other with low-fidelity time (≈25 years);
3.	Customers with a medium customer monetary value where most of their premiums go for Motor (≈70%), followed by Health
(17%). Here we see a main differentiator on the two customers’ profiles here present:
  a)	One with high-fidelity time (≈36 years);
  b)	Other with low-fidelity time (≈25 years);
4.	Customers with a medium customer monetary value where most of their premiums go for Household (≈44%), followed by 
Motor (≈22%) and Health (≈18%). Here we see a main differentiator on the two customers’ profiles here present:
  a)	One with high-fidelity time (≈36 years);
  b)	Other with low-fidelity time (≈25 years);
