"""Text processing for KickCast (Track A: Text / NLP).

Cleans the injury news from the FPL feed with our own code (lowercase,
strip punctuation, tokenize, remove stop words), converts it to TF-IDF
features and classifies each player's availability as fit, doubtful or
out.

Planned contents:
    clean_text()             - own text cleaner.
    AvailabilityClassifier   - TF-IDF vectorizer plus classifier.
"""