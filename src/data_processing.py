from typing import List, Dict
from collections import Counter
import torch
import csv


try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object

    examples = []

    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split("\t")
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue  # Skip malformed lines

            text, label_str = parts
            try:
                label = int(label_str)  # Convert label to integer
            except ValueError:
                print(f"Skipping line with invalid label: {line}")
                continue  # Skip lines with invalid labels

            examples.append(SentimentExample(tokenize(text), label))

    return examples

def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}  # Inicializar como un diccionario vacío
    index = 0

    for example in examples:
        for word in example.words:
            if word not in vocab:
                vocab[word] = index
                index += 1

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full

    bow_vector = torch.zeros(len(vocab), dtype=torch.float32)

    for word in text:
        if word in vocab:  
            index = vocab[word]
            if binary:
                bow_vector[index] = 1  
            else:
                bow_vector[index] += 1  

    return bow_vector
