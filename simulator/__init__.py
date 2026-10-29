"""NeuroContinuum-1 simulator package.

A from-scratch, brain-inspired (neuromorphic) spiking neural network
simulator.  It is both a research sandbox for the project's four novel
features and the executable *golden reference* for the RTL hardware in
``rtl/``.

Modules
-------
neurons       LIF & Izhikevich spike neuron populations
synapses      classic STDP connection model
plasticity    BCM meta-plasticity + reward-modulated STDP + homeostasis  (novel #1)
stochastic    noisy membrane for on-chip Monte-Carlo uncertainty          (novel #2)
temporal      latency encoding + first-spike race decoder                (novel #3)
selfhealing   stuck-synapse detection + automatic spare remapping        (novel #4)
stream        drifting seismic hazard waveform generator
domains       bridge / dam / factory structural-monitoring generators
encoding      rate / latency / poisson spike encodings
network       generic lock-step spiking network container
continuum_net the flagship end-to-end continual-learning classifier
"""

from .neurons import LIFLayer, IzhikevichLayer
from .synapses import STDPRule, Connection
from .plasticity import BCMPlasticity, RewardSTDP, SynapticHomeostasis
from .stochastic import StochasticLIFLayer, posterior_from_trials, decision_with_uncertainty
from .temporal import latency_encode, FirstSpikeDecoder
from .selfhealing import SelfHealingSynapseArray, FaultReport
from .stream import HazardStream, featurise as hazard_featurise, CLASS_NAMES
from .domains import DomainStream, DOMAINS, featurise
from .encoding import poisson_spikes, rate_to_current, pattern_input, spikes_to_rate
from .network import SNNetwork, SpikeRecord
from .continuum_net import NeuroContinuum, DriftTracker

__all__ = [
    "LIFLayer", "IzhikevichLayer",
    "STDPRule", "Connection",
    "BCMPlasticity", "RewardSTDP", "SynapticHomeostasis",
    "StochasticLIFLayer", "posterior_from_trials", "decision_with_uncertainty",
    "latency_encode", "FirstSpikeDecoder",
    "SelfHealingSynapseArray", "FaultReport",
    "HazardStream", "hazard_featurise", "CLASS_NAMES",
    "DomainStream", "DOMAINS", "featurise",
    "poisson_spikes", "rate_to_current", "pattern_input", "spikes_to_rate",
    "SNNetwork", "SpikeRecord",
    "NeuroContinuum", "DriftTracker",
]
