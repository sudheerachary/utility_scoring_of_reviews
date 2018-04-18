# Utility Scoring of Product Reviews
  - Utility is the usefulness or reliability of a given review.
  - Utility score is the score given to a review based on its usefulness from 0 to 1.
  
  - ## Dataset
    - We downloaded reviews and related data in 22 different domains like Electronics, Clothing, Sports,Video games etc from [amazon product data](http://jmcauley.ucsd.edu/data/amazon/)
    
  - ## Classification
    - Classify a given review into one of the 22 categories using
    - **Neural Network Architecture**
      - Embedding layer with the size of vocabulary.
      - 1-D convolutional layer with 32 filters of size 3 with relu activation.
      - Max pooling 1-D layer with pool size 2
      - Fully connected layer with relu activation function.
      - Fully connected layer with 22 nodes and softmax classifier.
    
    - **Voted Classifier**
      - Ensemble of 5 models (*SVM, Decision Tree, Naive Bayes* etc) to classify a given review using its **TF-IDF** vector.

    - **Adjectives**
      - Generated adjectives for each category using a seed set and recursively adding all its synonyms. The category with maximum count is the class of the given review.
    
    - **Learning Algorithms**
      - Support Vector Regression **(SVR)**
      - Simple linear regression **(SLR)**
      
    - **Comparision**
    <img src="reports/Screenshot from 2018-04-19 00.40.55.png"> <img src="reports/Screenshot from 2018-04-19 00.42.01.png"> <img src="reports/Screenshot from 2018-04-19 00.42.19.png">
  
  - ## Conclusions
    - **Classification**
      - Accuracy in classification is highest in the case of Voted Classifier. 
      - Voted classifier has the advantage of testing a review with 5 different classifiers.
      - Neural networks also works good but less accurate than voted classifier.
      - The naive method of counting number of matches in adjectives does not work well as there is no learning based in this method and also it is ambiguous
