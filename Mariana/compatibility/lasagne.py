from collections import OrderedDict
import Mariana.layers as ML
import Mariana.custom_types as MTYPES
import lasagne

__all__=["LasagneLayer", "LasagneStreamedLayer", "IAmAnnoyed"]

class IAmAnnoyed(Exception) :
    """What Mariana raises when you annoy her"""
    def __init__(self, msg) :
        self.message = msg

    def __str__(self) :
        return self.message
        
    def __repr__(self):
        return self.message

class LasagneStreamedLayer(object):
    """Wraps lasagne layers to give them a stream interface"""
    def __init__(self, incomingShape, streams, lasagneLayerCls, hyperParameters, initParameters, lasagneKwargs={}):
        super(LasagneStreamedLayer, self).__init__()
        self.streams = streams
        self.lasagneLayer = OrderedDict()

        kwargs = {}
        kwargs.update(hyperParameters)
        kwargs.update(initParameters)
        kwargs.update(lasagneKwargs)

        self.parameters = {}
        for f in streams :
            if len(self.parameters) == 0 :
                self.lasagneLayer[f] = lasagneLayerCls(incoming = incomingShape, **kwargs)
                for k in initParameters :
                    self.parameters[k] = getattr(self.lasagneLayer[f], k)
                kwargs.update(self.parameters)
            else :
                self.lasagneLayer[f] = lasagneLayerCls(incoming = incomingShape, **kwargs)

    def __getitem__(self, k) :
        return self.lasagneLayer[k]

    def __setitem__(self, k, v) :
        self.lasagneLayer[k] = v

class LasagneLayer(ML.Layer_ABC) :
    """This very special class allows you to incorporate a Lasagne layer seemlessly inside a Mariana network.
    An incorporated lasagne is just like a regular layer, with streams and all the other Mariana niceties.
    initializations must be specified with Mariana initializers, and please don't pass it an 'incoming', 'nonlinearity' argument.
    It is Mariana's job to do the shape inference and activate the layers, and she can get pretty upset if you try to tell her how to do her job.
    If you need to specifiy a specific value for some paramters, use the HardSet() initializer.

    Here's an examples::

        from lasagne.layers.dense import DenseLayer

        hidden = LasagneLayer(
            DenseLayer,
            initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
            lasagneHyperParameters={"num_units": 10},
            activation = MA.Tanh(),
            learningScenari = [MS.GradientDescent(lr = 0.1, momentum=0)],
            name = "HiddenLayer2"
        )
    
    """

    def __init__(self, lasagneLayerCls, initializations, lasagneHyperParameters={}, lasagneKwargs={}, **kwargs) :
        import inspect

        super(LasagneLayer, self).__init__(initializations=initializations, **kwargs)

        self.lasagneLayerCls = lasagneLayerCls

        self.lasagneHyperParameters = lasagneHyperParameters
        
        if "nonlinearity" in self.lasagneHyperParameters :
            raise IAmAnnoyed("There's an 'nonlinearity' argument in the hyperParameters. Use activation = <...>. Just like you would do for any other layer.")
        
        if "incoming" in self.lasagneHyperParameters :
            raise IAmAnnoyed("There's an 'incoming' argument in the hyperParameters. Don't tell me how to do my job!")
        
        self.addHyperParameters(self.lasagneHyperParameters)
        if "nonlinearity" in inspect.getargspec(lasagneLayerCls.__init__)[0] :
            self.lasagneHyperParameters["nonlinearity"] = None
        self.lasagneKwargs = lasagneKwargs
        
        self.lasagneLayer  = None
        # self.inLayer = None
        self.lasagneParameters = {}
        
        for init in self.abstractions["initializations"] :
            self.setP(init.getHP("parameter"), MTYPES.Parameter("%s.%s" % (self.name, init.getHP("parameter"))))
            init.setup(self)
            self.lasagneParameters[init.getHP("parameter")] = init.run

    def setShape_abs(self) :
        inLayer = self.network.getInConnections(self)[0]
        if not self.lasagneLayer and inLayer.getShape_abs() is not None:
            self.lasagneLayer = LasagneStreamedLayer(incomingShape=inLayer.getShape_abs(), streams=self.streams, lasagneLayerCls=self.lasagneLayerCls, hyperParameters=self.lasagneHyperParameters, initParameters=self.lasagneParameters)

    def femaleConnect(self, layer) :
        # self.inLayer = layer
        self.setShape_abs()

    def getShape_abs(self) :
        try :
            inLayer = self.network.getInConnections(self)[0]
        except IndexError :
            return None
        return self.lasagneLayer[self.streams[0]].get_output_shape_for(inLayer.getShape_abs())

    def _initParameters(self) :
        # self._setShape(self.inLayer)
        self.lasagneParameters = None
        for k, v in self.lasagneLayer.parameters.iteritems() :
            self.parameters[k].setValue(v, forceCast=False)

    def getParameterShape_abs(self, k) :
        v = getattr(self.lasagneLayer[self.streams[0]], k)
        return v.get_value().shape
    
    def setOutputs_abs(self) :
        inLayer = self.network.getInConnections(self)[0]
        for f in self.outputs.streams :
            self.outputs[f] = self.lasagneLayer[f].get_output_for(inLayer.outputs[f])
