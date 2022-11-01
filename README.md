# US-Housing-Rent-Prediction

As international or domestic students, many students arrive from various locations and countries, making it difficult for them to estimate the rent they must pay. Moreover, each student has unique requirements and options. As a result, their calculations for budgeting and spending could be challenging. Since housing is one of the most important basic requirements, we decided to do an analysis that will be useful for the current students as well as future cohorts in determining the proper rent price that is most suitable for them in order to attend the university. Each student has a unique background and specific preference. This is why we chose this dataset, which included characteristics such as the number of bedrooms, bathrooms, furnished options, laundry, parking options, pets allowed, smoking allowed or not, wheelchair accessibility, or even electric vehicle charging capabilities. We believe that these characteristics could be the factor contributing to the rental price. 

Questions To Investigate: 

1.	Which residential type is the most suitable to rent based on the individual requirements and amenities? (Logistic Regression)
2.	According to the students' individual needs and amenities, which price is the most suitable? (Random Forest Regression and XgBoost Regression)
3.	With respect to price and amenities, which region to expect? In addition, which features play an important role in prediction. (Random Forest Classifier)
4.	What is the expected number of beds based on the region and amenities? (Random Forest Classifier)


Conclusion
After conducting our descriptive and predictive analyses in depth, The following are concluding and scope for improvement considerations:

●	The best model for predicting the price amongst Linear Regression, Random Forest and XgBoost Regressor is random forest regression with an RMSE of 0.18, whereas linear regression is the worst model for the price prediction amongst the three.

●	For the logistic regression, the factor that has the most influence in altering the residential type is wheelchair access, followed by cat and dog allowed. On the other hand, the number of bathrooms, parking spaces, electrical outlets, and furnished options are features that have the least effect on altering to apartment type.

●	For the city prediction, the most important feature for our prediction is the price, and the least important feature is dining nearby, while the most important feature that affects the number of bedrooms is the region and the least is the dining nearby. 


# For more information, Kindly refer project report, jupyter notebook and script file.
