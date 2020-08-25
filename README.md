# A simple BiLSTM-CRF model for Chinese Named Entity Recognition

This repository includes the code for buliding a very simple __character-based BiLSTM-CRF sequence labeling model__ for Chinese Named Entity Recognition task. Its goal is to recognize three types of Named Entity: PERSON, LOCATION and ORGANIZATION.

This code works on __Python 3 & TensorFlow 1.2__ and the following repository [https:__github.com_guillaumegenthial_sequence_tagging](https:__github.com_guillaumegenthial_sequence_tagging) gives me much help.

## Model

This model is similar to the models provided by paper [1] and [2]. Its structure looks just like the following illustration:

![Network](._pics_pic1.png)

For one Chinese sentence, each character in this sentence has _ will have a tag which belongs to the set {O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}.

The first layer, __look-up layer__, aims at transforming each character representation from one-hot vector into *character embedding*. In this code I initialize the embedding matrix randomly. We could add some linguistic knowledge later. For example, do tokenization and use pre-trained word-level embedding, then augment character embedding with the corresponding token's word embedding. In addition, we can get the character embedding by combining low-level features (please see paper[2]'s section 4.1 and paper[3]'s section 3.3 for more details).

The second layer, __BiLSTM layer__, can efficiently use *both past and future* input information and extract features automatically.

The third layer, __CRF layer__,  labels the tag for each character in one sentence. If we use a Softmax layer for labeling, we might get ungrammatic tag sequences beacuse the Softmax layer labels each position independently. We know that 'I-LOC' cannot follow 'B-PER' but Softmax doesn't know. Compared to Softmax, a CRF layer can use *sentence-level tag information* and model the transition behavior of each two different tags.

## Dataset

|    | #sentence | #PER | #LOC | #ORG |
| :----: | :---: | :---: | :---: | :---: |
| train  | 46364 | 17615 | 36517 | 20571 |
| test   | 4365  | 1973  | 2877  | 1331  |

It looks like a portion of [MSRA corpus](http:__sighan.cs.uchicago.edu_bakeoff2006_). I downloaded the dataset from the link in `._data_path_original_link.txt`

### data files

The directory `._data_path` contains:

- the preprocessed data files, `train_data` and `test_data` 
- a vocabulary file `word2id.pkl` that maps each character to a unique id  

For generating vocabulary file, please refer to the code in `data.py`. 

### data format

Each data file should be in the following format:

```
中	B-LOC
国	I-LOC
很	O
大	O

句	O
子	O
结	O
束	O
是	O
空	O
行	O

```

If you want to use your own dataset, please: 

- transform your corpus to the above format
- generate a new vocabulary file

## How to Run

### train

`python main.py --mode=train `

### test

`python main.py --mode=test --demo_model=1521112368`

Please set the parameter `--demo_model` to the model that you want to test. `1521112368` is the model trained by me. 

An official evaluation tool for computing metrics: [here (click 'Instructions')](http:__sighan.cs.uchicago.edu_bakeoff2006_)

My test performance:

| P     | R     | F     | F (PER)| F (LOC)| F (ORG)|
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.8945 | 0.8752 | 0.8847 | 0.8688 | 0.9118 | 0.8515

### demo

`python main.py --mode=demo --demo_model=1521112368`

You can input one Chinese sentence and the model will return the recognition result:

![demo_pic](._pics_pic2.png)

## Reference

\[1\] [Bidirectional LSTM-CRF Models for Sequence Tagging](https:__arxiv.org_pdf_1508.01991v1.pdf)

\[2\] [Neural Architectures for Named Entity Recognition](http:__aclweb.org_anthology_N16-1030)

\[3\] [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](https:__link.springer.com_chapter_10.1007_978-3-319-50496-4_20)

\[4\] [https:__github.com_guillaumegenthial_sequence_tagging](https:__github.com_guillaumegenthial_sequence_tagging)  


tensor_name:  train_step/words/_word_embeddings/Adam_1
tensor_name:  train_step/words/_word_embeddings/Adam
tensor_name:  train_step/transitions/Adam
tensor_name:  train_step/proj/b/Adam
tensor_name:  train_step/proj/W/Adam_1
tensor_name:  bi-lstm/bidirectional_rnn/fw/lstm_cell/bias
tensor_name:  train_step/global_step
tensor_name:  train_step/bi-lstm/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
tensor_name:  train_step/proj/b/Adam_1
tensor_name:  train_step/bi-lstm/bidirectional_rnn/bw/lstm_cell/bias/Adam_1
tensor_name:  train_step/bi-lstm/bidirectional_rnn/fw/lstm_cell/kernel/Adam
tensor_name:  proj/b
tensor_name:  train_step/bi-lstm/bidirectional_rnn/bw/lstm_cell/kernel/Adam
tensor_name:  train_step/beta1_power
tensor_name:  proj/W
tensor_name:  transitions
tensor_name:  train_step/beta2_power
tensor_name:  train_step/transitions/Adam_1
tensor_name:  train_step/bi-lstm/bidirectional_rnn/fw/lstm_cell/bias/Adam_1
tensor_name:  train_step/bi-lstm/bidirectional_rnn/fw/lstm_cell/bias/Adam
tensor_name:  bi-lstm/bidirectional_rnn/fw/lstm_cell/kernel
tensor_name:  train_step/proj/W/Adam
tensor_name:  train_step/bi-lstm/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1
tensor_name:  bi-lstm/bidirectional_rnn/bw/lstm_cell/kernel
tensor_name:  train_step/bi-lstm/bidirectional_rnn/bw/lstm_cell/bias/Adam
tensor_name:  words/_word_embeddings
tensor_name:  bi-lstm/bidirectional_rnn/bw/lstm_cell/bias



tensor_name:  bi-lstm/bidirectional_rnn/fw/lstm_cell/bias
tensor_name:  proj/b
tensor_name:  proj/W
tensor_name:  transitions
tensor_name:  bi-lstm/bidirectional_rnn/fw/lstm_cell/kernel
tensor_name:  bi-lstm/bidirectional_rnn/bw/lstm_cell/kernel
tensor_name:  words/_word_embeddings
tensor_name:  bi-lstm/bidirectional_rnn/bw/lstm_cell/bias