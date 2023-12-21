#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving


# In[ ]:


errors = {
    0 : {"total": 0, 1: 0, 2: 0, 3: 0, 4: 0},
    1 : {"total": 0, 0: 0, 2: 0, 3: 0, 4: 0},
    2 : {"total": 0, 0: 0, 1: 0, 3: 0, 4: 0},
    3 : {"total": 0, 0: 0, 1: 0, 2: 0, 4: 0},
    4 : {"total": 0, 0: 0, 1: 0, 2: 0, 3: 0},
}
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0


# In[ ]:


loaded_model = tf.keras.saving.load_model('novel-forecast.keras')


# In[ ]:


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_errors = tf.strings.regex_replace(stripped_html, '[image error] ', '')
    return tf.strings.regex_replace(stripped_errors, '\r\n', '')


# In[ ]:


test_ds = keras.utils.text_dataset_from_directory(".\Reviews\\Test", batch_size=64, label_mode='int')
test_ds = test_ds.map(lambda x, y: (custom_standardization(x), y))


# In[ ]:


def make_prediction(model, text):
    review_arr = numpy.array([text])
    res = model(review_arr)
    max_index = numpy.argmax(res)
    return max_index


# In[ ]:


def is_negative(score):
    negative_scores = [0,1,2]
    return score in negative_scores


# In[ ]:


def check_review(text, realScore):
    global trueNegative, falsePositive, truePositive, falseNegative, errors
    predictedScore = make_prediction(loaded_model, text)
    
    if realScore != predictedScore:
        errors[realScore]['total'] = errors[realScore]['total'] + 1
        errors[realScore][predictedScore] = errors[realScore][predictedScore] + 1
    
    if is_negative(realScore) and is_negative(predictedScore):
        trueNegative = trueNegative + 1
    elif is_negative(realScore) and not is_negative(predictedScore):
        falsePositive = falsePositive + 1
    elif not is_negative(realScore) and not is_negative(predictedScore):
        truePositive = truePositive + 1
    else:
        falseNegative = falseNegative + 1


# In[ ]:


for element in test_ds:
    current_length = len(element[0].numpy());
    for i in range(current_length - 1):
        check_review(element[0].numpy()[i].decode("utf-8").strip(), int(element[1].numpy()[i]))

print('all done')


# In[ ]:


print(truePositive)


# In[ ]:


print(trueNegative)


# In[ ]:


print(falsePositive)


# In[ ]:


print(falseNegative)


# In[ ]:


print(errors)

