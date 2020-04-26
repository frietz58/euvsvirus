# Kalman Fake News Filter
## EuVsVirus Hackathon submission by "Friends of Kalman"

### What it does
Building on an initial assumption that the majority of fake news articles use similar writing patterns and buzz-words, we develop a machine-learning based approach with a neural classifier to distinguish between trustworthy and fake articles. Our web-server (http://134.100.14.116/) takes the entire text of the article as input and then tells you whether it appears to be real or fake after running our checker.
To do this, we first amass a fake news dataset with 30000+ articles that are verified as either trustworthy or fake, by combining previously published news classification datasets. We then process the data with GloVe, a pre-trained word embedding for distributed word representation, and then feed the vectorized articles to a neural classifier. 
Based on our current training and testing efforts, our model can accurately detect the truth or falsity of articles ~80% of the time.

### How we built it
We use 3 publicly available datasets:
- [Fake News: Kaggle dataset](https://www.kaggle.com/c/fake-news/data)
- [Liar: a benchmark dataset for fake news detection](https://github.com/thiagorainmaker77/liar_dataset)
- [Buzzfeed Political News Data & Random Political News Data](https://github.com/rpitrust/fakenewsdata1)

Each of the dataset entries is first preprocessed and then used for training a deep neural network classifier. The preprocessing consists of tokenization with [Keras](https://keras.io/preprocessing/text/) and vectorization with the [GloVe word embedding](https://nlp.stanford.edu/projects/glove/). The neural network classifier consists of an LSTM and 2 fully connected layers. It outputs a probability to which the input can be considered fake.

After training the network, it is used to classify the text that can be entered by users to evaluate through our website.

### What's next for Kalman Fake News Filter
We want to experiment with a more sophisticated classifier, namely the transformer-based model [BERT](https://arxiv.org/pdf/1810.04805.pdf). As setting up and this model takes significantly longer than the architecture we deployed, we could not integrate it before the deadline. We are eager to continue working on this project, so a comparison between models will be one of the next steps.

Moreover, we would like to make the “magic” of neural networks more intuitively understandable. For this, we will experiment with different types of visualization for the classifier and explore different ways to make our model more explainable. Further, it would be interesting to explore what the networks learned, aka which general patterns appear to be consistent for “fake news”. This could help with understanding the key characteristics of “fake news” and thus allow for easier identification of these on sight.

Finally, our AI is not perfect, therefore we want to connect the website to hand-checked fact-checking websites. We believe that a mix of hand-checked facts and AI checked facts will deliver the best results overall. The website can use the hand-checked facts (that we know are true) and make inferences on facts that have not yet been checked due to the massive amount of information published every day.
