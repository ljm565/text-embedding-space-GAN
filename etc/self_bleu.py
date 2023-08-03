from fast_bleu import SelfBLEU



class Self_BLEU:
    def __init__(self, max_val):
        assert max_val >= 2
        self.weights = {i: tuple([1/i]*i) for i in range(2, max_val+1)}

    
    def get_sbleu(self, sentences):
        scores = SelfBLEU(sentences, self.weights)
        scores = scores.get_score()
        scores = {k: sum(v) / len(v) for k, v in scores.items()}
        return scores