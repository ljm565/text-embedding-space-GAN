class DataSynthesisRatio:
    def __init__(self, reference):
        self.reference = reference

    def get_dsr(self, sentences):
        repeat_num = 0
        for s in sentences:
            if s in self.reference and len(s.split()) > 10:
                repeat_num += 1

        dup, div = 1 - repeat_num / len(sentences), len(set(sentences)) / len(sentences)
        repeat_rate = (dup, div, dup * div, 2 * dup * div / (dup + div))
        return repeat_rate