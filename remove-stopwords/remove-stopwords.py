def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here*
    result = [token for token in tokens if token not in stopwords]
    return result