import theano
import theano.tensor as tt
from Mariana.abstraction import Abstraction_ABC

__all__ = ["Cost_ABC", "Null", "NegativeLogLikelihood", "MeanSquaredError", "CrossEntropy", "CategoricalCrossEntropy", "BinaryCrossEntropy", "VaeLL", "VaeKL"]

class Cost_ABC(Abstraction_ABC) :
    """This is the interface a Cost must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
    this class must also include a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters."""

    def apply(self, layer, targets, outputs, purpose) :
        """Apply to a layer and update networks's log. Purpose is supposed to be a sting such as 'train' or 'test'"""
        hyps = {}
        for k in self.hyperParameters :
            hyps[k] = getattr(self, k)
        
        message = "%s uses cost %s for %s" % (layer.name, self.__class__.__name__, purpose)
        layer.network.logLayerEvent(layer, message, hyps)
        return self.costFct(targets, outputs)

    def costFct(self, targets, outputs) :
        """The cost function. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Null(Cost_ABC) :
    """No cost at all"""
    def costFct(self, targets, outputs) :
        return tt.sum(outputs*0 + targets*0)

class NegativeLogLikelihood(Cost_ABC) :
    """For a probalistic output, works great with a softmax output layer"""
    def costFct(self, targets, outputs) :
        cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
        return cost

class MeanSquaredError(Cost_ABC) :
    """The all time classic"""
    def costFct(self, targets, outputs) :
        cost = tt.mean((outputs - targets) ** 2)
        return cost

class CategoricalCrossEntropy(Cost_ABC) :
    """Returns the average number of bits needed to identify an event."""
    def costFct(self, targets, outputs) :
        cost = tt.mean( tt.nnet.categorical_crossentropy(outputs, targets) )
        return cost
        
class CrossEntropy(CategoricalCrossEntropy) :
    """Short hand for CategoricalCrossEntropy"""
    pass

class BinaryCrossEntropy(Cost_ABC) :
    """Use this one for binary data"""
    def costFct(self, targets, outputs) :
        cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
        return cost

class VaeGaussKL(Cost_ABC):
    """
        D_{kl}(q(z|x) || p(z)) for gaussian q(z|x) and p(z) = N(mu, sigma^2), with
        diagonal covariance.

        `mu`: vector-like - mean for p(z)
        `logsigma`: vector-like - log(sigma) for p(z)
    """

    def __init__(self, mu=None, logsigma=None):
        if mu is None:
            mu = 0
        if logsigma is None:
            logsigma = 0
        self.mup = mu
        self.logsigmap = logsigma

    def run(self, targets, outputs):
        mu, logsigma = tt.split(outputs, [outputs.shape[1] / 2, outputs.shape[1] / 2], 2, axis=1)
        return -0.5 * tt.mean(tt.sum(1 - tt.exp(2 * logsigma) / tt.exp(2 * self.logsigmap) - tt.sqr(self.mup - mu) / tt.exp(self.logsigmap) + 2 * logsigma - 2 * self.logsigmap, axis=1))

class VaeKL(Cost_ABC):
    """
        KL term for VAEs with p(z) and q(z|x) of the same family.
        `fn`: str - distribution family to use (only 'gaussian' supported).
    """

    def __init__(self, fn='gaussian', **kwargs):
        super(VaeKL, self).__init__(**kwargs)
        self.fn = fn
        self.kl = {'gaussian': VaeGaussKL}
        self.cost = self.kl[self.fn](**kwargs)

    def run(self, targets, outputs):
        return self.cost.run(targets, outputs)

class VaeGaussLL(Cost_ABC):
    """E_{q(z|x)}[log p(x|z)], single sample estimate, for gaussian p(x|z)"""

    def run(self, targets, outputs):
        mu, logsigma = tt.split(outputs, [outputs.shape[1] / 2, outputs.shape[1] / 2], 2, axis=1)
        return -tt.mean(tt.sum(-(numpy.float32(0.5 * numpy.log(2 * numpy.pi)) + logsigma) - 0.5 * tt.sqr(targets - mu) / tt.exp(2 * logsigma), axis=1))

class VaeBernLL(Cost_ABC):
    """E_{q(z|x)}[log p(x|z)], single sample estimate, for bernoulli vector p(x|z)"""

    def run(self, targets, outputs):
        cost = tt.mean(tt.sum(tt.nnet.binary_crossentropy(tt.minimum(1 - 1e-7, tt.maximum(outputs, 1e-07)), targets), axis=1))
        return cost

class VaeLL(Cost_ABC):
    """
        Expectation term for VAEs.
        `fn`: str - either 'gaussian' or 'bernoulli', indicating which distribution for p(x | z).
    """

    def __init__(self, fn='gaussian', **kwargs):
        super(VaeLL, self).__init__(**kwargs)
        self.fn = fn
        self.ll = {'gaussian': VaeGaussLL, 'bernoulli': VaeBernLL}
        self.cost = self.ll[self.fn](**kwargs)

    def run(self, targets, outputs):
        return self.cost.run(targets, outputs)


