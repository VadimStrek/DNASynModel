import pickle
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

class DNASynModel:
    def __init__(self, k=3, alphabet="ACGT", pseudocount=1.0):
        """
        Initialize a k-th order Markov model for DNA sequences.
        
        Parameters:
        k : int, default=3
            Order of the Markov model (length of the context).
        alphabet : str, default="ACGT"
            Alphabet of allowed symbols.
        pseudocount : float, default=1.0
            Pseudocount added to transition counts during probability
            estimation.
        """
        self.k = k
        self.alphabet = list(alphabet)
        self.pseudocount = pseudocount
        self.counts = defaultdict(Counter)
        self.probs = {}

    def fit(self, sequences):
        """
        Estimate transition probabilities from a dataset of DNA sequences.
        The model counts transitions from every k-mer context to the
        following nucleotide and converts the counts into conditional
        probabilities using Laplace smoothing.
        
        Parameters:
        sequences : iterable of str
            Dataset of DNA sequences.
        
        Returns:
        DNASynModel
            Trained model.
        """
        for seq in sequences:
            seq = seq.upper()
            if len(seq) <= self.k:
                continue
            for i in range(len(seq) - self.k):
                context = seq[i:i+self.k]
                nxt = seq[i+self.k]
                if nxt not in self.alphabet:
                    continue
                self.counts[context][nxt] += 1
        self._normalize()
        return self

    def _normalize(self):
        """
        Convert transition counts into conditional probabilities.
        """
        self.probs = {}
        # Add pseudocounts to avoid zero probabilities.
        for context, cnt in self.counts.items():
            total = sum(cnt.values()) + self.pseudocount * len(self.alphabet)
            self.probs[context] = {nt: (cnt[nt] + self.pseudocount) / total for nt in self.alphabet}

    def parameters(self):
        """
        Create the transition probability matrix.
        
        Returns:
        pandas.DataFrame
            DataFrame whose rows correspond to Markov contexts and columns
            correspond to nucleotides from the alphabet.
        """
        df = pd.DataFrame.from_dict(self.probs, orient="index")
        df = df[self.alphabet]
        df.index.name = "context"
        return df.sort_index()

    def generate(self, length, seed=None):
        """
        Generate a synthetic DNA sequence.
        The first context is selected randomly unless an initial seed is
        provided. Each subsequent nucleotide is sampled according to the
        transition probabilities of the current context.
        
        Parameters:
        length : int
            Length of the generated sequence.
        seed : str or None, default=None
            Initial context of length k. If None, a random context is used.
        
        Returns:
        str
            Generated DNA sequence.
        """
        if seed is None:
            context = random.choice(list(self.probs.keys()))
        else:
            if len(seed) != self.k:
                raise ValueError(f"Seed length must equal k={self.k}")
            context = seed.upper()
        sequence = list(context)
        while len(sequence) < length:
            if context not in self.probs:
                context = random.choice(list(self.probs.keys()))
            probs = self.probs[context]
            nt = random.choices(population=self.alphabet,
                                weights=[probs[a] for a in self.alphabet],
                                k=1)[0]
            sequence.append(nt)
            context = "".join(sequence[-self.k:])
        return "".join(sequence)

    def sample(self, n_sequences, length):
        """
        Generate sample of synthetic DNA sequences.
        
        Parameters:
        n_sequences : int
            Number of sequences to generate.
        length : int
            Length of each generated sequence.
        
        Returns:
        list of str
            Generated DNA sequences.
        """
        return [self.generate(length) for _ in range(n_sequences)]

    def log_likelihood(self, sequence):
        """
        Compute the log-likelihood of a DNA sequence.
        The likelihood is computed as the sum of logarithms of conditional
        transition probabilities defined by the Markov model.
        
        Parameters:
        sequence : str
            DNA sequence.
        
        Returns:
        float
            Log-likelihood of the sequence.
        """
        sequence = sequence.upper()
        logp = 0.0
        # Use a uniform probability for unseen contexts.
        default = 1 / len(self.alphabet)
        for i in range(len(sequence) - self.k):
            context = sequence[i:i+self.k]
            nxt = sequence[i+self.k]
            if context in self.probs:
                p = self.probs[context].get(nxt, default)
            else:
                p = default
            logp += np.log(p)
        return logp

    def evaluate(self, sequences):
        """
        Evaluate the model on a dataset of DNA sequences.
        The method computes standard quality metrics, including
        log-likelihood, negative log-likelihood (NLL), perplexity,
        information criteria (AIC and BIC), and average information
        content in bits per nucleotide.
        
        Parameters:
        sequences : iterable of str
            Dataset of DNA sequences.
        
        Returns:
        dict
            Dictionary containing model evaluation metrics.
        """
        total_logp = 0
        total_symbols = 0
        for seq in sequences:
            if len(seq) <= self.k:
                continue
            total_logp += self.log_likelihood(seq)
            total_symbols += len(seq)-self.k
        nll = -total_logp / total_symbols
        perplexity = np.exp(nll)
        p = len(self.probs) * (len(self.alphabet)-1)
        n = total_symbols
        aic = 2*p - 2*total_logp
        bic = np.log(n)*p - 2*total_logp
        bits_per_nt = nll / np.log(2)
        return {"order": self.k,
                "contexts": len(self.probs),
                "parameters": p,
                "observations": n,
                "log_likelihood": total_logp,
                "NLL": nll,
                "Perplexity": perplexity,
                "AIC": aic,
                "BIC": bic,
                "bits_per_nt": bits_per_nt}

    def save(self, filename):
        """
        Save the trained model to a file.
        The model order, alphabet, pseudocount and transition
        probabilities are stored to reconstruct the model.
        
        Parameters:
        filename : str
            Output file name.
        """
        state = {"k": self.k,
                "alphabet": self.alphabet,
                "pseudocount": self.pseudocount,
                "probs": self.probs}
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename):
        """
        Load a previously saved model.
        
        Parameters:
        filename : str
            Path to a serialized model.
        
        Returns:
        DNASynModel
            Restored Markov model.
        """
        with open(filename, "rb") as f:
            state = pickle.load(f)
        model = cls(k=state["k"],
                    alphabet="".join(state["alphabet"]),
                    pseudocount=state["pseudocount"])
        model.probs = state["probs"]
        model.counts = defaultdict(Counter)
        return model
