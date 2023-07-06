# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:53:35 2023

@author: semih
"""

###############################################################################
# LIBRARIES
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from snowballstemmer import TurkishStemmer

from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline



import matplotlib.pyplot as plt

nltk.download("amw-1.4")
nltk.download("wordnet")
###############################################################################

###############################################################################
# STOPWORDS
stopwords = stopwords.words("turkish")
###############################################################################


###############################################################################
# NORMALIZATION START

data_train = pd.read_csv("Turkish_Sentiment_Analysis/train.csv")
data_test = pd.read_csv("Turkish_Sentiment_Analysis/test.csv")

data_all = pd.concat([data_train, data_test])
data_all.drop(labels="dataset", axis=1, inplace=True)


########
# Lemma işlemi sonucu oluşan yeni veri seti
data_all = pd.read_csv("data2_with_labels.csv")


data_all["text"] = data_all["text"].apply(str) # Burada text sütunundaki verileri string yapıyoruz. Aksi halde x.split ile böldüğümüz alanda float64 türünün split fonksiyonunun olmadığı şeklinde hata veriyor. Tüm ifadeyi string yapıp o şekilde ilerliyoruz.

data_all["text"] = data_all["text"].apply(lambda x: " ".join(word.lower() for word in x.split()))
data_all["text"] = data_all["text"].str.replace("[^a-zA-Zçğıöşü]"," ") # Türkçe karakterler ve harfler haricindeki her şeyi kaldırıyoruz. (Sayılar, özel karakterler vs.)

data_all["text"] = data_all["text"].apply(lambda x: " ".join(word for word in x.split() if word not in stopwords)) # içindeki stopwrds kelimeleri barındıran kelimeleri de siliyoruz.

data_all["text"] = data_all["text"].apply(lambda x: " ".join(word for word in x.split() if word != "unk"))
data_all["text"] = data_all["text"].apply(lambda x: " ".join(word for word in x.split() if not word.startswith("http") or word.startswith("https")))
data_all["text"] = data_all["text"].apply(lambda x: " ".join(word for word in x.split() if word.find("https") == -1 or word.find("http") == -1))
data_all["text"] = data_all["text"].apply(lambda x: " ".join(kelime for kelime in x.split() if len(kelime) > 1)) # lemma işlemi sonunda 1 harflik kelimeleri siliyoruz.
#######


###############################################################################
# Positive : 262166
# Nötr     : 170917
# Negative : 56561
print(data_all["label"].value_counts()) 

###############################################################################
# Data visualation

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.title("Ürün Yorumları Durum Grafiği")
plt.bar(["Positive", "Notr" , "Negative"], data_all["label"].value_counts())
plt.show()


def Graph_Pie(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)

###############################################################################


###############################################################################
# STEMMER START
lemma = WordNetLemmatizer() # Bu şekilde kelimenin sıfat, zarf, isim olması gibi durumlarını anlayabiliyoruz. Arka planda büyük bir dil bilgisi kuralları var.
lemma_ = data_all["text"].apply(lambda x: " ".join(lemma.lemmatize(word) for word in x.split()))

"""
ps = PorterStemmer()
ps_ = data_all["text"].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))


snowballstemmer = TurkishStemmer()
ts_ = data_all["text"].apply(lambda x: " ".join(snowballstemmer.stemWord(word) for word in x.split()))
"""


# PorterStemmer, TurkishStemmer ve lemma işlemleri arasından en iyi çalışan lemma gibi görünüyor. Bu sebeple lemma ile devam edilmeye karar verilmiştir. 
# STEMMER END
###############################################################################

data_all["text"] = data_all["text"].apply(lambda x: " ".join(lemma.lemmatize(word) for word in x.split()))


data_all["text"] = data_all["text"].apply(lambda x: " ".join(word for word in x.split() if len(word)>1)) # 1 kelimeden az olan cümleleri almıyoruz.

data_all["text"] = data_all["text"].replace("^\s*$", np.nan, regex=True) # boş satırları nan yapıyoruz.
data_all.dropna(axis=0, inplace=True) # nan yaptığımız boş satırları siliyoruz.
data_all.drop_duplicates(inplace=True) # aynı olan satırları siliyoruz.
# NORMALIZATION END
###############################################################################

###############################################################################
# TRANSFORMATIZATION
encoder = preprocessing.LabelEncoder()
data_all["label"] = encoder.fit_transform(data_all["label"])
data_all.reset_index(drop=True, inplace=True)


###############################################################################
# Veri ön işleme sonucu ürün yorumlarının durumu
# Positive : 252196
# Nötr     : 167880
# Negative : 55561
print(data_all["label"].value_counts()) 

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.title("Veri Ön İşleme Sonucu Ürün Yorumlarının Durumu")
plt.bar(["Positive", "Nötr" , "Negative"], data_all["label"].value_counts())
plt.show()


model = Pipeline([('vctrzr', TfidfVectorizer()),('clf',naive_bayes.MultinomialNB())])
#model = Pipeline([('vctrzr', CountVectorizer(max_features=1000, binary=True)),('clf',naive_bayes.MultinomialNB())])
#model = Pipeline([('vctrzr', CountVectorizer(max_features=1000, ngram_range=(1,2))),('clf',naive_bayes.MultinomialNB())])
#model = Pipeline([('vctrzr', CountVectorizer(max_features=1000)),('clf',naive_bayes.MultinomialNB())])

#model = Pipeline([('vctrzr', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),('clf',naive_bayes.MultinomialNB())])

from sklearn.neighbors import KNeighborsClassifier
#model = Pipeline([('vctrzr', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),('clf',KNeighborsClassifier())])

"""



"""


print(pd.Series(data_all["text"]).value_counts)
print(pd.Series(data_all["label"]).value_counts)



seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
Accuracy_Scores = []
Recall_Scores = []
F1_Scores = []
Precision_Scores = []
All_Confusion_Matrix = []

nb = naive_bayes.MultinomialNB()

for train, test in kfold.split(data_all["text"], data_all["label"]):
    model.fit(data_all["text"][train], y=data_all["label"][train])
    predicted = model.predict(data_all["text"][test]) 
    
    train_x, train_y, test_x, test_y = data_all["text"][train], data_all["label"][train], data_all["text"][test], data_all["label"][test]

    Accuracy_Scores.append(accuracy_score(data_all["label"][test],predicted))
    Precision_Scores.append(precision_score(data_all["label"][test], predicted, average='weighted'))
    Recall_Scores.append(recall_score(data_all["label"][test],predicted, average='weighted'))
    F1_Scores.append(f1_score(data_all["label"][test],predicted, average='weighted'))
    All_Confusion_Matrix.append(confusion_matrix(data_all["label"][test], predicted))



print(train_y.value_counts()) 
print(test_y.value_counts())


plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.title("Eğitim Verisi Ürün Yorumlarının Durumu")
plt.bar(["Positive", "Nötr" , "Negative"], train_y.value_counts())
plt.show()


plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.title("Test Verisi Ürün Yorumlarının Durumu")
plt.bar(["Positive", "Nötr" , "Negative"], test_y.value_counts())
plt.show()


myexplode = [0.2, 0, 0, 0]
plt.title("Tüm Veri Pasta Grafiği")
plt.pie(data_all["label"].value_counts(), labels=["Positive", "Notr", "Negative"],autopct='%1.1f%%')
plt.show()

plt.title("Eğitim Veri Seti Durum Grafiği")
plt.pie(train_y.value_counts(), labels=["Positive", "Notr", "Negative"],autopct='%1.1f%%')
plt.show()

plt.title("Test Veri Seti Durum Grafiği")
plt.pie(test_y.value_counts(), labels=["Positive", "Notr", "Negative"],autopct='%1.1f%%')
plt.show()


print(f" Mean Accuracy Score: {np.mean(Accuracy_Scores)}\n Mean Recall Score: {np.mean(Recall_Scores)}\n Mean Precision Score: {np.mean(Precision_Scores)}\n Mean F1 Score: {np.mean(F1_Scores)}\n Mean Confusion Matrix: {np.mean(All_Confusion_Matrix)} ")


print(pd.Series(data=data_all["label"]).value_counts())
print(pd.Series(data_all["text"][test]).value_counts())




pred_data = ['seni seviyorum', "sana aşığım", "bugün hava çok güzel", "yarın hava çok sıcak olacak", "çok sıkıldım", "onu sevip sevmediğimi bilmiyorum", "emin değilim", "sevdim", "görmedim"]
d3 = pd.DataFrame(pred_data, columns=['text'])
model.fit(data_all["text"][train], y=data_all["label"][train])


tfid = TfidfVectorizer()
tfid.fit(train_x)

train_x_tf = tfid.transform(train_x)
test_x_tf = tfid.transform(d3)
nb.fit(train_x_tf, train_y)
preds = nb.predict(tfid.transform(test_x))
res = nb.predict(tfid.transform(pd.Series("Keşke gitsem")))


knn_ = KNeighborsClassifier()
knn_.fit(train_x_tf, train_y)
res_knn = knn_.predict(tfid.transform(pd.Series("senden nefret ediyorum")))



allVocab = tfid.get_feature_names()




word_list = pd.Series(" ".join(data_all["text"]).split())
counted_words = word_list.value_counts().sort_values()

"""
import zeyrek
nltk.download('punkt')
analyzer = zeyrek.MorphAnalyzer()
ps = pd.Series()
for ex in d3["text"]:
    #print(ex)
    result = analyzer.analyze(ex)
    for word_result in result:
       print(word_result[0].lemma)
        
                
           
    



for word_result in result:
   print(word_result[0].word)
        
   for parse in word_result:
       print(parse.formatted)
        
       print()
       
"""
#result = analyzer.analyze("olta")






nbb = naive_bayes.MultinomialNB()
nbb.fit(train_x_tf, train_y)
print(nbb.predict(tfid.transform(d3)))
    

"""
nb.fit(train_x_tf, train_y)
newDocument = vectorizerCount.transform(pd.Series("seni seviyorum"))
result = nb.predict(newDocument) 
"""


##########1#####################################################################







"""
for train, test in kfold.split(data_all["text"], data_all["label"]):
    #X_train = vectorizerCount.fit_transform(df["text"][train]).toarray()
    #X_test = vectorizerCount.transform(df["text"][test]).toarray()
    #nb.fit(X_train, y=df["label"][train])
    #predicted = nb.predict(X_test)
    
    # Pipeline ile farklı modeller kullandığımız için burada direkt modeli fit ve predict edebiliriz. diğer türlü bu kısmı sürekli değiştirmemiz gerekecekti.
    model.fit(data_all["text"][train], y=data_all["label"][train])
    predicted = model.predict(data_all["text"][test]) 
    
    Accuracy_Scores.append(accuracy_score(data_all["label"][test],predicted))
    Precision_Scores.append(precision_score(data_all["label"][test], predicted))
    Recall_Scores.append(recall_score(data_all["label"][test],predicted))
    F1_Scores.append(f1_score(data_all["label"][test],predicted))
    All_Confusion_Matrix.append(confusion_matrix(data_all["label"][test], predicted))


print("Average Accuracy Score"+str(np.mean(Accuracy_Scores)))
print("Average Precision Score"+str(np.mean(Precision_Scores)))
print("Average Recall Score"+str(np.mean(Recall_Scores)))
print("Average F1 Score"+str(np.mean(F1_Scores)))
print(f"Average Confusion Matrix Score: {np.mean(All_Confusion_Matrix)}")

"""




















