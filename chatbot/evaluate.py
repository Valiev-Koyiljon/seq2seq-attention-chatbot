import torch
from utils import normalizeString
from config import MAX_LENGTH

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    """Evaluate an input sentence and generate its response"""
    # Format input sentence as a batch
    indexes_batch = [voc.indexesFromSentence(sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(searcher.device)
    lengths = lengths.to("cpu")

    # Run through searcher
    tokens, scores = searcher(input_batch, lengths, max_length)

    # Decode tokens to words
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    """Evaluate input from the user"""
    encoder.eval()
    decoder.eval()
    
    # Start chat loop
    print("Let's chat! (type 'q' or 'quit' to exit)")
    while True:
        try:
            # Get input
            input_sentence = input('> ')
            # Check if it's time to quit
            if input_sentence.lower() in ['q', 'quit']:
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")