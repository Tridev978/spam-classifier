import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,f1_score,ConfusionMatrixDisplay

df=pd.read_csv("data/spam.csv",encoding="latin-1")[["v1","v2"]]
df.columns=["label","message"]
df["label"]=df["label"].map({"ham":0,"spam":1})
df=df.drop_duplicates()

X_train,X_test,Y_train,Y_test=train_test_split(
    df["message"],df["label"],
    test_size=0.2,random_state=42,stratify=df["label"]
)

models= {
    "Naive Bayes":MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}
best_name,best_pipe,best_f1,best_preds=None,None,0,None
for name,clf in models.items():
    pipe=Pipeline([
        ("tfidf",TfidfVectorizer(lowercase=True,stop_words="english")),
        ("clf",clf),
    ])
    pipe.fit(X_train,Y_train)
    preds=pipe.predict(X_test)
    f1=f1_score(Y_test,preds)
    print(f"==={name}===")
    print(classification_report(Y_test,preds,target_names=["ham","spam"]))
    if f1>best_f1:
        best_name,best_pipe,best_f1,best_preds=name,pipe,f1,preds

print("Best Model",best_name)
ConfusionMatrixDisplay.from_predictions(Y_test,best_preds,display_labels=["ham","spam"])
plt.title(best_name)
plt.savefig("confusion_matrix.png")
joblib.dump(best_pipe,"model/spam_model.joblib")
print("Model saved.")