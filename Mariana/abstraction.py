from collections import OrderedDict
import Mariana.custom_types as MTYPES

__all__ = ["Logger_ABC", "Abstraction_ABC", "UntrainableAbstraction_ABC", "TrainableAbstraction_ABC", "Apply_ABC"]

class Logger_ABC(object):
    """Interface for objects that log events"""
    def __init__(self, **kwargs):
        super(Logger_ABC, self).__init__()
        self.log = []
        self.notes = OrderedDict()

    def logEvent(self, message, **moreStuff) :
        """log an event"""
        import time
            
        entry = {
            "date": time.ctime(),
            "timestamp": time.time(),
            "message": message,
        }
        entry.update(moreStuff)
        self.log.append(entry)

    def getLog(self) :
        """return the log"""
        return self.log

    def addNote(self, title, text) :
        """add a note"""
        self.notes[title] = text

    def printLog(self) :
        """JSON pretty printed log"""
        import json
        print json.dumps(self.getLog(), indent=4, sort_keys=True)

class Abstraction_ABC(Logger_ABC):
    """
    This class represents a layer modifier. This class must includes a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, streams=["train", "test"], **kwargs):
        super(Abstraction_ABC, self).__init__()
        
        self.streams = streams
        self.hyperParameters = OrderedDict()

        self._mustInit=True

    def isTrainable(self) :
        raise NotImplementedError("Must be implemented in child")

    def getParameters(self) :
        raise NotImplementedError("Must be implemented in child")

    def addHyperParameters(self, dct) :
        """adds to the list of hyper params, dct must be a dict"""
        self.hyperParameters.update(dct)

    def setHyperparameter(k, v) :
        """sets a single hyper parameter"""
        self.setHP(k, v)

    def getHyperparameter(k, v) :
        """get a single hyper parameter"""
        self.getHP(k, v)

    def setHP(self, k, v) :
        """setHyperparameter() alias"""
        self.hyperParameters[k] = v

    def getHP(self, k) :
        """getHyperparameter() alias"""
        return self.hyperParameters[k]

    def getHyperParameters(k, v) :
        """return all hyper parameter"""
        return self.hyperParameters

    def toDictionary(self) :
        """A dct representation of the object"""
        res = {
            "name": str(self.name),
            "hyperParameters": OrderedDict(self.hyperParameters),
            "notes": OrderedDict(self.notes),
            "class": self.__class__.__name__
        }
        
        return res

    def __repr__(self) :
        return "< %s, hps: %s >" % (self.__class__.__name__, dict(self.hyperParameters))

class UntrainableAbstraction_ABC(Abstraction_ABC):
    """docstring for UntrainableAbstraction_ABC"""
    
    def isTrainable(self) :
        return False

    def getParameters(self) :
        return {}

    def __hash__(self) :
        return hash(self.__class__.__name__ + str(self.hyperParameters.items()))
        
    def __eq__(self, a) :
        return self.__class__ is a.__class__ and self.hyperParameters == a.hyperParameters

class TrainableAbstraction_ABC(Abstraction_ABC):
    """docstring for TrainableAbstraction_ABC"""
    def __init__(self, initializations=[], learningScenari=[], regularizations=[], **kwargs):
        super(TrainableAbstraction_ABC, self).__init__(**kwargs)

        self.abstractions={
            "initializations": initializations,
            "learningScenari": learningScenari,
            "regularizations": regularizations,
        }

        self.parameters = OrderedDict()
        # self._mustInit = True

    def getAbstractions(self) :
        res = []
        for absts in self.abstractions.itervalues() :
            for ab in absts :
                res.append(ab)
        return res

    def isTrainable(self) :
        return True

    def setParameter(k, v) :
        """Brutally set the value of a parameter. No checks applied"""
        self.setP(k, v)
    
    def setParameters(self, dct) :
        """to set a bunch of them with a dict"""
        for k, v in dct.iteritems() :
            self.setP(k, v)

    def getParameter(self, k) :
        """get a single parameter"""
        self.getP(k)

    def setP(self, param, value) :
        """setParameter() alias"""
        if isinstance(value, MTYPES.Parameter) :
            self.parameters[param] = value
        else :
            self.parameters[param].setValue(value)
    
    def getP(self, k) :
        """getParameter() alias"""
        return self.parameters[k]

    def getParameters(self) :
        """return all parameter"""
        return self.parameters

    def hasParameter(self, p) :
        """returns wether I have parameter called p"""
        return p in self.parameters

    def hasP(self, p) :
        """hasParameter() alias"""
        return self.hasParameter(p)

    def _getParameterShape_abs(self, param) :
        if param not in self.parameters :
            raise ValueError("Unknown parameter: %s for %s" % (param, self))
        return self.getParameterShape_abs(param)

    def getParameterShape_abs(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplementedError("Should be implemented in child")

    def _parametersSanityCheck(self) :
        "perform basic parameter checks on layers, automatically called on initialization"
        for k, v in self.getParameters().iteritems() :
            if not v.isSet() :
                raise ValueError("Parameter '%s' of '%s' has not been initialized" % (k, self.name) )

    def _initParameters(self, forceReset=False) :
        """creates the parameters if necessary"""
        selfParams = set()
        for k, v in self.parameters.iteritems() :
            if not v.isSet() :
                selfParams.add(k)

        initParams = set()
        if self._mustInit or forceReset :
            for init in self.abstractions["initializations"] :
                if not self.getP(init.getHP("parameter")).isSet() :
                    init._apply(self)
                    initParams.add(init.getHP("parameter"))
        # print len(selfParams), len(initParams)
        # print self, selfParams, initParams
        if len(selfParams) != len(initParams) :
            raise ValueError("Parameters: %s of %s, where not supplied initializations" % ((selfParams - initParams), self) )
        
        self._mustInit=False

    def toDictionary(self) :
        """A dct representation of the object"""
        
        res = super(TrainableAbstraction_ABC, self).toDictionary()
        ps = OrderedDict()
        for k, v in self.parameters.iteritems() :
            ps[k] = {"shape": self.getParameterShape_abs(k)}

        res["parameters"] = ps    
        
        return res

    def __repr__(self) :
        return "< %s, hps: %s, ps: %s >" % (self.__class__.__name__, self.hyperParameters, self.parameters)

class Apply_ABC(object):
    """Interface for abstractions that are applyied to other abstractions (all but layers)"""

    def __init__(self, **kwargs):
        super(Apply_ABC, self).__init__()
        self.name = self.__class__.__name__
        self._mustInit=True
        # self.parent = None
    
    # def setParent(self, layer) :
    #     message = "'%s' has been claimed by '%s'" % (self.name, layer.name)
    #     self.logEvent(message)
    #     self.parent = layer

    # def unsetParent(self) :
    #     message = "'%s' is once again unclaimed" % (self.name)
    #     self.logEvent(message)
    #     self.parent = None

    def logApply(self, layer, **kwargs) :
        message = "Applying : '%s' on layer '%s'" % (self.name, layer.name)
        self.logEvent(message)
    
    def setup(self, abstraction) :
        """setups the untrainable abstraction for a given trainable abstraction. Called just before apply(). By default does nothing, but allows some degree of fine tuning if necesssary"""
        pass

    def _apply(self, layer, **kwargs) :
        """Logs and calls self.apply()"""
        self.setup(layer)
        self.logApply(layer, **kwargs)
        self.apply(layer, **kwargs)

    def apply(self, layer, **kwargs) :
        """Apply to a layer, basically logs stuff and then calls run"""
        raise NotImplementedError("Must be implemented in child")
    
    def run(self, **kwargs) :
        """the actual worker function that does whaters the abstraction is supposed to do"""
        raise NotImplementedError("Must be implemented in child")
