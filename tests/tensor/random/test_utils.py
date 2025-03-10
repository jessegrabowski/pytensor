import numpy as np
import pytest

from pytensor import config, function
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor.random.utils import (
    RandomStream,
    broadcast_params,
    supp_shape_from_ref_param_shape,
)
from pytensor.tensor.type import matrix, tensor
from tests import unittest_tools as utt


@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    rewrites_query = RewriteDatabaseQuery(include=[None], exclude=[])
    py_mode = Mode("py", rewrites_query)
    with config.change_flags(mode=py_mode, compute_test_value="warn"):
        yield


def test_broadcast_params():
    ndims_params = [0, 0]

    mean = np.array([0, 1, 2])
    cov = np.array(1e-6)
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], mean)
    assert np.array_equal(res[1], np.broadcast_to(cov, (3,)))

    ndims_params = [1, 2]

    mean = np.r_[1, 2, 3]
    cov = np.stack([np.eye(3) * 1e-5, np.eye(3) * 1e-4])
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], np.broadcast_to(mean, (2, 3)))
    assert np.array_equal(res[1], cov)

    mean = np.stack([np.r_[0, 0, 0], np.r_[1, 1, 1]])
    cov = np.arange(3 * 3).reshape((3, 3))
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], mean)
    assert np.array_equal(res[1], np.broadcast_to(cov, (2, 3, 3)))

    mean = np.stack([np.r_[0, 0, 0], np.r_[1, 1, 1]])
    cov = np.stack(
        [np.arange(3 * 3).reshape((3, 3)), np.arange(3 * 3).reshape((3, 3)) * 10]
    )
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], mean)
    assert np.array_equal(res[1], cov)

    mean = np.array([[1, 2, 3]])
    cov = np.stack([np.eye(3) * 1e-5, np.eye(3) * 1e-4])
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], np.array([[1, 2, 3], [1, 2, 3]]))
    assert np.array_equal(res[1], cov)

    mean = np.array([[0], [10], [100]])
    cov = np.diag(np.array([1e-6]))
    params = [mean, cov]
    res = broadcast_params(params, ndims_params)
    assert np.array_equal(res[0], mean)
    assert np.array_equal(res[1], np.broadcast_to(cov, (3, 1, 1)))

    # Try it in PyTensor
    with config.change_flags(compute_test_value="raise"):
        mean = tensor(dtype=config.floatX, shape=(None, 1))
        mean.tag.test_value = np.array([[0], [10], [100]], dtype=config.floatX)
        cov = matrix()
        cov.tag.test_value = np.diag(np.array([1e-6], dtype=config.floatX))
        params = [mean, cov]
        res = broadcast_params(params, ndims_params)
        assert np.array_equal(res[0].get_test_value(), mean.get_test_value())
        assert np.array_equal(
            res[1].get_test_value(), np.broadcast_to(cov.get_test_value(), (3, 1, 1))
        )


class TestSharedRandomStream:
    def test_tutorial(self):
        srng = RandomStream(seed=234)
        rv_u = srng.uniform(0, 1, size=(2, 2))
        rv_n = srng.normal(0, 1, size=(2, 2))

        f = function([], rv_u)
        # Disabling `default_updates` means that we have to pass
        # `srng.state_updates` to `function` manually, if we want the shared
        # state to change
        g = function([], rv_n, no_default_updates=True)
        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

        assert np.all(f() != f())
        assert np.all(g() == g())
        assert np.all(abs(nearly_zeros()) < 1e-5)

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_basics(self, rng_ctor):
        random = RandomStream(seed=utt.fetch_seed(), rng_ctor=rng_ctor)

        with pytest.raises(ValueError):
            random.uniform(0, 1, size=(2, 2), rng=np.random.default_rng(23))

        with pytest.raises(AttributeError):
            random.blah

        assert hasattr(random, "standard_normal")

        with pytest.raises(AttributeError):
            np_random = RandomStream(namespace=np.random, rng_ctor=rng_ctor)
            np_random.ndarray

        fn = function([], random.uniform(0, 1, size=(2, 2)), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = np.random.SeedSequence(utt.fetch_seed())
        (rng_seed,) = rng_seed.spawn(1)
        rng = random.rng_ctor(rng_seed)

        numpy_val0 = rng.uniform(0, 1, size=(2, 2))
        numpy_val1 = rng.uniform(0, 1, size=(2, 2))

        assert np.allclose(fn_val0, numpy_val0)
        assert np.allclose(fn_val1, numpy_val1)

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_seed(self, rng_ctor):
        init_seed = 234
        random = RandomStream(init_seed, rng_ctor=rng_ctor)

        assert random.default_instance_seed == init_seed

        new_seed = 43298
        random.seed(new_seed)

        rng_seed = np.random.SeedSequence(new_seed)
        assert random.gen_seedgen.entropy == rng_seed.entropy

        random.seed()

        rng_seed = np.random.SeedSequence(init_seed)
        assert random.gen_seedgen.entropy == rng_seed.entropy

        # Reset the seed
        random.seed(new_seed)

        # Check state updates
        _ = random.normal()

        # Now, change the seed when there are state updates
        random.seed(new_seed)

        update_seed = np.random.SeedSequence(new_seed)
        (update_seed,) = update_seed.spawn(1)
        ref_rng = random.rng_ctor(update_seed)
        state_rng = random.state_updates[0][0].get_value(borrow=True)

        if hasattr(state_rng, "get_state"):
            random_state = state_rng.get_state()

            # hack to try to get something reasonable for ref_rng
            try:
                ref_state = ref_rng.get_state()
            except AttributeError:
                ref_state = list(ref_rng.bit_generator.state.values())

            assert np.array_equal(random_state[1], ref_state[1])
            assert random_state[0] == ref_state[0]
            assert random_state[2:] == ref_state[2:]
        else:
            ref_state = ref_rng.bit_generator.state
            random_state = state_rng.bit_generator.state
            assert random_state["bit_generator"] == ref_state["bit_generator"]
            assert random_state["state"] == ref_state["state"]

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_uniform(self, rng_ctor):
        # Test that RandomStream.uniform generates the same results as numpy
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        fn = function([], random.uniform(-1, 1, size=(2, 2)))
        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = np.random.SeedSequence(utt.fetch_seed())
        (rng_seed,) = rng_seed.spawn(1)

        rng = random.rng_ctor(rng_seed)
        numpy_val0 = rng.uniform(-1, 1, size=(2, 2))
        numpy_val1 = rng.uniform(-1, 1, size=(2, 2))

        assert np.allclose(fn_val0, numpy_val0)
        assert np.allclose(fn_val1, numpy_val1)

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_default_updates(self, rng_ctor):
        # Basic case: default_updates
        random_a = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        out_a = random_a.uniform(0, 1, size=(2, 2))
        fn_a = function([], out_a)
        fn_a_val0 = fn_a()
        fn_a_val1 = fn_a()
        assert not np.all(fn_a_val0 == fn_a_val1)

        nearly_zeros = function([], out_a + out_a - 2 * out_a)
        assert np.all(abs(nearly_zeros()) < 1e-5)

        # Explicit updates #1
        random_b = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        out_b = random_b.uniform(0, 1, size=(2, 2))
        fn_b = function([], out_b, updates=random_b.updates())
        fn_b_val0 = fn_b()
        fn_b_val1 = fn_b()
        assert np.all(fn_b_val0 == fn_a_val0)
        assert np.all(fn_b_val1 == fn_a_val1)

        # Explicit updates #2
        random_c = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        out_c = random_c.uniform(0, 1, size=(2, 2))
        fn_c = function([], out_c, updates=random_c.state_updates)
        fn_c_val0 = fn_c()
        fn_c_val1 = fn_c()
        assert np.all(fn_c_val0 == fn_a_val0)
        assert np.all(fn_c_val1 == fn_a_val1)

        # No updates at all
        random_d = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        out_d = random_d.uniform(0, 1, size=(2, 2))
        fn_d = function([], out_d, no_default_updates=True)
        fn_d_val0 = fn_d()
        fn_d_val1 = fn_d()
        assert np.all(fn_d_val0 == fn_a_val0)
        assert np.all(fn_d_val1 == fn_d_val0)

        # No updates for out
        random_e = RandomStream(utt.fetch_seed(), rng_ctor=rng_ctor)
        out_e = random_e.uniform(0, 1, size=(2, 2))
        fn_e = function([], out_e, no_default_updates=[random_e.state_updates[0][0]])
        fn_e_val0 = fn_e()
        fn_e_val1 = fn_e()
        assert np.all(fn_e_val0 == fn_a_val0)
        assert np.all(fn_e_val1 == fn_e_val0)

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_multiple_rng_aliasing(self, rng_ctor):
        # Test that when we have multiple random number generators, we do not alias
        # the state_updates member. `state_updates` can be useful when attempting to
        # copy the (random) state between two similar pytensor graphs. The test is
        # meant to detect a previous bug where state_updates was initialized as a
        # class-attribute, instead of the __init__ function.

        rng1 = RandomStream(1234, rng_ctor=rng_ctor)
        rng2 = RandomStream(2392, rng_ctor=rng_ctor)
        assert rng1.state_updates is not rng2.state_updates
        assert rng1.gen_seedgen is not rng2.gen_seedgen

    @pytest.mark.parametrize("rng_ctor", [np.random.default_rng])
    def test_random_state_transfer(self, rng_ctor):
        # Test that random state can be transferred from one pytensor graph to another.

        class Graph:
            def __init__(self, seed=123):
                self.rng = RandomStream(seed, rng_ctor=rng_ctor)
                self.y = self.rng.uniform(0, 1, size=(1,))

        g1 = Graph(seed=123)
        f1 = function([], g1.y)
        g2 = Graph(seed=987)
        f2 = function([], g2.y)

        for su1, su2 in zip(g1.rng.state_updates, g2.rng.state_updates, strict=True):
            su2[0].set_value(su1[0].get_value())

        np.testing.assert_array_almost_equal(f1(), f2(), decimal=6)


def test_supp_shape_from_ref_param_shape():
    with pytest.raises(ValueError, match="^ndim_supp*"):
        supp_shape_from_ref_param_shape(
            ndim_supp=0,
            dist_params=(np.array([1, 2]), 0),
            ref_param_idx=0,
        )

    res = supp_shape_from_ref_param_shape(
        ndim_supp=1,
        dist_params=(np.array([1, 2]), np.eye(2)),
        ref_param_idx=0,
    )
    assert res == (2,)

    res = supp_shape_from_ref_param_shape(
        ndim_supp=1,
        dist_params=(np.array([1, 2]), 0),
        param_shapes=((2,), ()),
        ref_param_idx=0,
    )
    assert res == (2,)

    with pytest.raises(ValueError, match="^Reference parameter*"):
        supp_shape_from_ref_param_shape(
            ndim_supp=1,
            dist_params=(np.array(1),),
            ref_param_idx=0,
        )

    res = supp_shape_from_ref_param_shape(
        ndim_supp=2,
        dist_params=(np.array([1, 2]), np.ones((2, 3, 4))),
        ref_param_idx=1,
    )
    assert res == (3, 4)

    res = supp_shape_from_ref_param_shape(
        ndim_supp=2,
        dist_params=(np.array([1, 2]), np.ones((2, 3, 4))),
        param_shapes=((2,), (2, 3, 4)),
        ref_param_idx=1,
    )
    assert res == (3, 4)
