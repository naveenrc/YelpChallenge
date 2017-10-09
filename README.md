### Yelp Dataset challenge
This project is about Natural Language processing and classifying images obtained from yelp dataset which can be downloaded from here https://www.yelp.com/dataset.<br>

### Requirements
NLP
* python 3.5
* spacy
* gensim
* pyLDAvis
* Word2vec
* Bokeh
* tSNE

Image classification
* python 3.5
* tensorflow
* opencv
* 12 GB RAM

### NLP
Modern_NLP.ipynb walks through the following topics(best viewed on nb viewer)
1. A tour of the dataset
2. Introduction to text processing with spaCy
3. Automatic phrase modeling
4. Topic modeling with LDA
5. Visualizing topic models with pyLDAvis
6. Word vector models with word2vec
7. Visualizing word2vec with t-SNE


### Setup for image classification
1. Install project requirements
2. Create a folder yelpData and move the extracted data from yelp into this folder.
3. Run photo_process.py to transform the original images to the size you desire. Enter a photo size when prompted.
4. Run photo_info.py to get information about the photos
5. Run classifier.py to start the model (may take longer, 6 to 10 hours without a GPU)
6. Run predict.py to predict image label
```bash
pip install -r requirements.txt
python ./photoAnalysis/photo_process.py
python ./photoAnalysis/photo_info.py
python ./classifier/classifier.py
python ./classifier/predict.py
```

<a href="https://drive.google.com/file/d/0BypHvhe9eW_KcWFDa3pyZlNJams/view?usp=sharing">Report for image classification can be found here</a>
