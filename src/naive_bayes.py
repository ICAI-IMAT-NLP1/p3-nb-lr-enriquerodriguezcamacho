import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)  # Estimar priors
        self.vocab_size = features.shape[1]  # Definir vocab_size
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)


    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        num_samples = labels.shape[0]  
        class_counts = torch.bincount(labels.to(torch.int64))  
        class_priors = {i: count / num_samples for i, count in enumerate(class_counts)}

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        num_classes = int(torch.max(labels).item()) + 1  
        vocab_size = features.shape[1]  

        word_counts = {c: torch.zeros(vocab_size) for c in range(num_classes)}
        
        for c in range(num_classes):
            class_features = features[labels == c]  
            word_counts[c] = class_features.sum(dim=0)  

        conditional_probs = {
            c: (word_counts[c] + delta) / (word_counts[c].sum() + delta * vocab_size)
            for c in range(num_classes)
        }

        return conditional_probs
    

    def estimate_class_posteriors(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError("Model must be trained before estimating class posteriors.")

        log_posteriors = {}

        for c in self.class_priors:
            log_prior = torch.log(self.class_priors[c])  # Log of the prior probability
            log_likelihood = torch.sum(feature * torch.log(self.conditional_probabilities[c]))  # Sum log probabilities of words
            log_posteriors[c] = log_prior + log_likelihood  # Naive Bayes formula in log space

        return torch.tensor([log_posteriors[0], log_posteriors[1]])
    
    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        
        log_posteriors = self.estimate_class_posteriors(feature)  

        pred = torch.argmax(log_posteriors).item() 
        
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)  # Compute log probabilities

        probs = torch.nn.functional.softmax(log_posteriors, dim=0)  # Convert log probabilities to normalized probabilities

        return probs
