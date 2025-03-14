import numpy as np
import pytest

from pytensor import config, function
from pytensor.d3viz.formatting import PyDotFormatter
from pytensor.printing import _try_pydot_import


try:
    _try_pydot_import()
except Exception as e:
    pytest.skip(f"pydot not available: {e!s}", allow_module_level=True)

from tests.d3viz import models


class TestPyDotFormatter:
    def setup_method(self):
        self.rng = np.random.default_rng(0)

    def node_counts(self, graph):
        node_types = [node.get_attributes()["node_type"] for node in graph.get_nodes()]
        a, b = np.unique(node_types, return_counts=True)
        nc = dict(zip(a, b, strict=True))
        return nc

    @pytest.mark.parametrize("mode", ["FAST_RUN", "FAST_COMPILE"])
    def test_mlp(self, mode):
        m = models.Mlp()
        f = function(m.inputs, m.outputs, mode=mode)
        pdf = PyDotFormatter()
        graph = pdf(f)
        if mode == "FAST_RUN":
            expected = 13
        elif mode == "FAST_COMPILE":
            expected = 14
        assert len(graph.get_nodes()) == expected
        nc = self.node_counts(graph)

        if mode == "FAST_RUN":
            assert nc["apply"] == 7
        elif mode == "FAST_COMPILE":
            assert nc["apply"] == 8
        assert nc["output"] == 1

    def test_ofg(self):
        m = models.Ofg()
        f = function(m.inputs, m.outputs)
        pdf = PyDotFormatter()
        graph = pdf(f)
        assert len(graph.get_nodes()) == 10
        sub_graphs = graph.get_subgraph_list()
        assert len(sub_graphs) == 2
        ofg1, ofg2 = sub_graphs
        if config.mode == "FAST_COMPILE":
            assert len(ofg1.get_nodes()) == 8
        else:
            assert len(ofg1.get_nodes()) == 5
        assert len(ofg1.get_nodes()) == len(ofg2.get_nodes())

    def test_ofg_nested(self):
        m = models.OfgNested()
        f = function(m.inputs, m.outputs)
        pdf = PyDotFormatter()
        graph = pdf(f)
        assert len(graph.get_nodes()) == 7
        assert len(graph.get_subgraph_list()) == 1
        ofg1 = graph.get_subgraph_list()[0]
        assert len(ofg1.get_nodes()) == 6
        assert len(ofg1.get_subgraph_list()) == 1
        ofg2 = ofg1.get_subgraph_list()[0]
        assert len(ofg2.get_nodes()) == 4
