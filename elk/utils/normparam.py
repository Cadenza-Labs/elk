
# TODO : Maybe you can use something like this to integrate into the CcsReporter for evaluation time
# Class to store global normalisation parameters to be used at evaluation time
class NormParam:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

class ContrastNormParam:
    def __init__(self, posParam: NormParam, negParam: NormParam):
        self.pos: NormParam = posParam
        self.neg: NormParam = negParam