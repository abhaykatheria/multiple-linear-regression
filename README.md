# multiple-linear-regression
predicting the profit of 50 startups depending on their r&amp;d spend location etc

After linear regression i decided to make multple linear regression which make prediction by learning from different multiple feature.
The following project uses the dataset of  50 startup and we have to predict their profits.
The Code has been made in such way that it can be tweaked to work for other models as well
I have used pandas to read dataset and numpy for vectorisation 
Sklearn is used to generate test and train data 
Their is one categorical feature in the data so i have encoded it using one hot encoder 
Normalisation was really necessary in ths case as data values were really large.
I needed to avoid Dummy variable trap that is why deleted the first column genereated by One hot encoder
