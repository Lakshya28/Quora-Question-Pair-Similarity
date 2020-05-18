# Quora-Question-Pair-Similarity
<p align="center">
  <img width="800" height="450" src="https://cdn.vox-cdn.com/thumbor/-G6pZqbvh3j1ttYuUpOyehb-yCs=/0x28:640x388/1600x900/cdn.vox-cdn.com/assets/1296846/quoralogo.jpg">
</p>


Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

## Problem Statement:

Identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not.

### Some Basic Analysis

BOW and TF-IDF are two of the most common methods people use in information retrieval. Generally speaking, SVMs and Naive Bayes
are more common for classification problem, however, because their accuracy is dependent of the training data, Xgboost provided
the best accuracy in this particular data set. XGBoost is a gradient boosting framework that has become massively popular, especially
in the Kaggle community.
* Not to remove stop words, because words like “what”, “which” and “how” may have strong signals.
* Not to stem words.
* Remove punctuation.
* Correct typos.
* Change abbreviations to its original terms.
* Remove comma between numbers.
* Change special chars to words. And so on.

### Feature Extraction:
* **Basic Features - Extracted some features before cleaning of data as below.**

1. **freq_qid1** = Frequency of qid1's
2. **freq_qid2** = Frequency of qid2's
3. **q1len** = Length of q1
4. **q2len** = Length of q2
5. **q1_n_words** = Number of words in Question 1
6. **q2_n_words** = Number of words in Question 2
7. **word_Common** = (Number of common unique words in Question 1 and Question 2)
8. **word_Total** =(Total num of words in Question 1 + Total num of words in Question 2)
9. **word_share** = (word_common)/(word_Total)
10. **freq_q1+freq_q2** = sum total of frequency of qid1 and qid2
11. **freq_q1-freq_q2** = absolute difference of frequency of qid1 and qid2

* **Advanced Features**
1. **cwc_min** = common_word_count / (min(len(q1_words), len(q2_words))
2. **cwc_max** = common_word_count / (max(len(q1_words), len(q2_words))
3. **csc_min** = common_stop_count / (min(len(q1_stops), len(q2_stops))
4. **csc_max** = common_stop_count / (max(len(q1_stops), len(q2_stops))
5. **ctc_min** = common_token_count / (min(len(q1_tokens), len(q2_tokens))
6. **ctc_max** = common_token_count / (max(len(q1_tokens), len(q2_tokens))
7. **last_word_eq** = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
8. **first_word_eq** = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
9. **abs_len_diff** = abs(len(q1_tokens) - len(q2_tokens))
10. **mean_len** = (len(q1_tokens) + len(q2_tokens))/2
11. **fuzz_ratio** = How much percentage these two strings are similar, measured with edit distance.[More Detail...](https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/)
12. **fuzz_partial_ratio** = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.[More Detail...](https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/)
13. **token_sort_ratio** = sorting the tokens in string and then scoring fuzz_ratio.[More Detail...](https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/)
14. **longest_substr_ratio** = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

## Word Embeddings
Word Embeddings are the texts converted into numbers and there may be different numerical representations of the same text.A very basic definition of a word embedding is a real number, vector representation of a word. Typically, these days, words with similar meaning will have vector representations that are close together in the embedding space.The beauty of representing words as vectors is that they lend themselves to mathematical operators. For example, we can add and subtract vectors — the canonical example here is showing that by using word vectors we can determine that:

**king — man + woman = queen**

In other words, we can subtract one meaning from the word vector for king (i.e. maleness), add another meaning (femaleness), and show that this new word vector (king — man + woman) maps most closely to the word vector for queen.

* Extracted Tf-Idf features for this combained question1 and question2 Train data.Transformed test data into same vector space.

* From Pretrained glove word vectors got average word vector for question1 and question2. With this avg word vector got below distances.
1. Cosine distance
2. Euclidean distance
3. Minkowski distance
4. Jaccard Distance

## Useful Links
* **TF-IDF**
  * https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63
  * https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
  * https://stevenloria.com/tf-idf/
  * http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
  * http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
* **Word2Vec and Embeddings**
  * https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf
  * https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc
  * https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
  * https://code.google.com/archive/p/word2vec/

