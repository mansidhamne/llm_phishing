<h1>Mail Phishing Detection</h1>
<p>Using LLM and various baseline models, we infer whether incoming mails are spam or not spam.</p>

This project will train a few baseline models and a ```RoBERTa``` model to perform a binary classification task on short texts. It will be structured into a training and inference pipeline.

## Dataset Used:

- [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)

  Size: 5729 examples
  | label | count |
  |---|---|
  | Spam (1) | 1368 |
  | Non Spam (0) | 4327 | 

## Models Used:
Baseline Models
  - [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
  - [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  - [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
  - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
  - [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

LLM Model:
  - [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta)

### Results:

Baseline Models:

|   | f1  | precision  |  recall |  accuracy | training_time  |  inference_time |
|---|---|---|---|---|---|---|
| NB |	0.9031007751937980 |	0.8503649635036500 | 0.962809917355372 |	0.9561018437225640 |	0.0048120021820068 |	0.0026149749755859
| LR | 0.9444444444444440 |	0.9306569343065690 |	0.9586466165413530 |	0.9736611062335380 |	0.0553030967712402 |	0.0003492832183837
| KNN |	0.8576512455516010 | 0.8795620437956200	| 0.8368055555555560 |	0.929762949956102	| 0.0007669925689697 |	0.1529159545898440
| SVM	| 0.9441441441441440 | 0.9562043795620440	| 0.9323843416370110 |	0.9727831431079890 | 0.6286451816558840	| 0.121741771697998
| XGBoost |	0.8783783783783780 | 0.948905109489051 | 0.8176100628930820 |	0.9367866549604920 | 1.163405179977420	| 0.0014581680297851600

RoBERTa:

|   | f1  | precision  |  recall |  accuracy | training_time  |  inference_time |
|---|---|---|---|---|---|---|
| RoBERTa |	0.9686924493554330 | 0.9776951672862450 |	0.9598540145985400 |	0.9850746268656720 | 14531.401168823200	| 293.3671679496770

From the above metrics we can see in baseline models, highest accuracy on this dataset is provided by **Logistic Regression** equal to **94.44%** while the **LLM model RoBERTa** provides **96.86%** accuracy.




