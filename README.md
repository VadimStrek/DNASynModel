# DNASynModel
A k-th order Markov model for synthetic DNA sequence generation.

The model estimates conditional nucleotide probabilities from a dataset of DNA sequences and can subsequently be used to
- generate synthetic DNA sequences;
- compute sequence log-likelihoods;
- evaluate model quality;
- save and restore trained models.

Transition probabilities are estimated using Laplace smoothing.
