import drjit as dr
import numpy as np

# utils
Float = dr.llvm.ad.Float


def get_grad(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropGrad | dr.ADFlag.ClearVertices)
    grad = dr.grad(a)
    dr.set_grad(a, 0)
    return grad


def get_sq_grad(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropVar | dr.ADFlag.ClearVertices)
    sq_sum = dr.grad(a)
    dr.set_grad(a, 0)
    return sq_sum


def get_ones(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropOnes | dr.ADFlag.ClearVertices)
    ones = dr.grad(a)
    dr.set_grad(a, 0)
    return ones

def get_grad_gard2_count(a, loss):
    dr.backward(loss, flags=dr.ADFlag.BackPropVarianceCounter | dr.ADFlag.ClearVertices)
    g = dr.graphviz_ad()
    g.view()
    grad = dr.grad(a)
    grad2 = dr.grad2(a)
    counter = dr.counter(a)
    dr.set_grad(a, 0)
    return grad, grad2, counter


def compute_variance(grad, sq_grad, ones):
    return sq_grad - grad * grad / ones


def assert_backprop(a, loss, expected_grad, expected_grad2, expected_counter):
    grad, grad2, counter = get_grad_gard2_count(a, loss)
    print(f"grad: {grad}")
    print(f"expected grad: {expected_grad}")
    print()
    print(f"grad2: {grad2}")
    print(f"expected grad2: {expected_grad2}")
    print()
    print(f"counter: {counter}")
    print(f"expected counter: {expected_counter}")
    print()

    assert np.allclose(grad, expected_grad)
    assert np.allclose(grad2, expected_grad2)
    assert np.allclose(counter, expected_counter)
    print("-------------------------------")


def assert_variance(a, loss, expected_var):
    grad, grad2, counter = get_grad_gard2_count(a, loss)
    print(grad, grad2, counter)
    var = compute_variance(grad, grad2, counter)
    assert var == expected_var


class Test1D:
    def test_basic(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return x

        def df(x):
            return 1

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)

    def test_linear(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return 2 * x

        def df(x):
            return 2

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)

    def test_sum(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return x + x

        def df(x):
            return 1 + 1

        def df_sq_sum(x):
            return 1**2 + 1**2

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 2.0)
        assert_variance(a, f(a), 0.0)

    def test_linear_array(self):
        a = Float(4.0)
        dr.enable_grad(a)

        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            b = x * Float(arr)
            return dr.sum(2.0 * b)

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_linear_array_2(self):
        a = Float(4.0)
        dr.enable_grad(a)
        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            return dr.sum(2.0 * x * Float(arr))

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_linear_array_3(self):
        a = Float(4.0)
        dr.enable_grad(a)
        arr = np.array([1.0, 2.0, 3.0, 4.0])

        def f(x):
            return dr.sum(2.0 * x * Float(arr) * 1.0)

        def df(x):
            return np.sum(2.0 * arr)

        def df_sq_sum(x):
            return np.sum((2.0 * arr) ** 2)

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 4.0)

    def test_product(self):
        a = Float(3.0)
        dr.enable_grad(a)

        def f(x):
            return x * x

        def df(x):
            return x + x

        def df_sq_sum(x):
            return x**2 + x**2

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 2.0)
        assert_variance(a, f(a), 0.0)

    def test_square(self):
        a = Float(3.0)
        dr.enable_grad(a)
        coeff = 2.00001

        def f(x):
            return x**coeff

        def df(x):
            return coeff * x ** (coeff - 1)

        assert_backprop(a, f(a), df(a), df(a) ** 2, 1.0)
        assert_variance(a, f(a), 0.0)


class TestND:
    def test_basic(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = a
        assert_backprop(a, loss, [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        assert_variance(a, loss, [0.0, 0.0])

    def test_linear(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = 2 * a
        assert_backprop(a, loss, [2.0, 2.0], [4.0, 4.0], [1.0, 1.0])
        assert_variance(a, loss, [0.0, 0.0])

    def test_sum(self):
        a = Float([4.0, 5.0])
        dr.enable_grad(a)
        loss = (a + a) * a
        assert_backprop(a, loss, [2.0, 2.0], [2.0, 2.0], [2.0, 2.0])
        assert_variance(a, loss, [0.0, 0.0])


if __name__ == "__main__":
    t1d = Test1D()
    tnd = TestND()
    # t1d.test_basic()
    # t1d.test_linear()
    # t1d.test_sum()
    # t1d.test_linear_array()
    # t1d.test_linear_array_2()
    # t1d.test_linear_array_3()
    # t1d.test_product()
    # t1d.test_square()
    # tnd.test_basic()
    # tnd.test_linear()
    tnd.test_sum()
