# Neural Conversational Chatbot with Attention

An implementation of a sequence-to-sequence neural chatbot with attention mechanisms using PyTorch.


```
Human: Hi there!
Bot: Hello! Welcome to our chat.
Human: How are you?
Bot: I am doing well. 
```

## Core Research Papers

### Sequence-to-Sequence Learning
1. **Neural Machine Translation by Jointly Learning to Align and Translate**
   - Authors: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
   - [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
   - *Key contribution: Introduced attention mechanisms for sequence-to-sequence models*

2. **Sequence to Sequence Learning with Neural Networks**
   - Authors: Ilya Sutskever, Oriol Vinyals, Quoc V. Le
   - [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
   - *Key contribution: Foundation of sequence-to-sequence architecture*

### Attention Mechanisms
3. **Effective Approaches to Attention-based Neural Machine Translation**
   - Authors: Minh-Thang Luong, Hieu Pham, Christopher D. Manning
   - [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
   - *Key contribution: Global and local attention mechanisms*

4. **Learning Phrase Representations using RNN Encoder-Decoder**
   - Authors: Kyunghyun Cho et al.
   - [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
   - *Key contribution: Introduced GRU cells*

### Conversational Models
5. **A Neural Conversational Model**
   - Authors: Oriol Vinyals, Quoc V. Le
   - [arXiv:1506.05869](https://arxiv.org/abs/1506.05869)
   - *Key contribution: Applied seq2seq to open-domain conversations*

## Project Structure
```
seq2seq-attention-chatbot/
├── data/
│   └── movie-corpus/
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── decoder.py 
│   ├── encoder.py
│   └── search.py
├── save/
├── __init__.py
├── config.py
├── evaluate.py
├── experiment.ipynb
├── train.py
├── utils.py
└── vocabulary.py
```

## Model Architecture
- Encoder: Bidirectional GRU 
- Attention: Luong's global attention
- Decoder: GRU with attention
- Parameters:
  - Hidden size: 500
  - Layers: 2 (both encoder/decoder)
  - Dropout: 0.1
  - Learning rate: 0.0001
  - Batch size: 64

## Dataset
Uses the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html):
- 220,579 conversation exchanges
- 304,713 utterances
- 9,035 characters
- 617 movies

## Implementation References
- Yuan-Kuei Wu's PyTorch chatbot implementation
- Sean Robertson's Sequence-to-sequence tutorial
- Matthew Inkawhich's chatbot tutorial

## License
MIT License

