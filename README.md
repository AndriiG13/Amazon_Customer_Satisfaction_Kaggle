# Amazon_Customer_Satisfaction_Kaggle

This repository contains R script to the in-class Kaggle activity on classifying customer reviews. This activity was part 
of the Behavioural Data Science course at the University of Amsterdam, carried out by me and Lukas Klima. 

We were given textual reviews and ratings given to baby products sold on Amazon. The goal was to build a model that could 
classify reviews into either the below 4 stars category or 4 stars and above category. 

To build the model, a sparse matrix with each word as a variable was first constructed. We then used ridge regression from 
the "glment" package to build the final model.

In the end we were able to classify reviews into the two categories with a 94% accuracy when tested on a test set.




