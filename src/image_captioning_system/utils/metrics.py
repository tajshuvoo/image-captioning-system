from nltk.translate.bleu_score import corpus_bleu


def compute_bleu_scores(references, predictions):
    """
    references: list[list[list[str]]]
    predictions: list[list[str]]
    """

    bleu_1 = corpus_bleu(
        references, predictions, weights=(1.0, 0, 0, 0)
    )
    bleu_2 = corpus_bleu(
        references, predictions, weights=(0.5, 0.5, 0, 0)
    )
    bleu_3 = corpus_bleu(
        references, predictions, weights=(0.33, 0.33, 0.33, 0)
    )
    bleu_4 = corpus_bleu(
        references, predictions, weights=(0.25, 0.25, 0.25, 0.25)
    )

    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4,
    }
