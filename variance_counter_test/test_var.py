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

def get_grad_gard2_count(a, loss, see_graph=True):
    dr.backward(loss, flags=dr.ADFlag.BackPropVarianceCounter | dr.ADFlag.ClearVertices)
    if see_graph:
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


def debug_backprop(a, loss):
    grad, grad2, counter = get_grad_gard2_count(a, loss)
    print(f"grad: {grad}")
    print(f"grad2: {grad2}")
    print(f"counter: {counter}")
    print("-------------------------------")


def assert_variance(a, loss, expected_var):
    grad, grad2, counter = get_grad_gard2_count(a, loss)
    print(grad, grad2, counter)
    var = compute_variance(grad, grad2, counter)
    print(f"var: {var}")
    # assert var == expected_var


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
    
    def test_basic2(self):
        a = Float(4.0)
        dr.enable_grad(a)

        def f(x):
            return x * 3

        def df(x):
            return 3

        assert_backprop(a, f(f(a)), 9,  81, 1.0)
    
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
    
    def test_add_product(self):
        a = Float(3.0)
        dr.enable_grad(a)

        def f(x):
            return (x + x) * x

        def df(x):
            return (x + x) + x *( 1 + 1 )

        def df_sq_sum(x):
            return (x + x)**2 + x**2 *( 1**2 + 1**2 )

        assert_backprop(a, f(a), df(a), df_sq_sum(a), 3.0)

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
        # loss = a + a

        assert_backprop(a, loss, [16.0, 20.0], [96.0, 150.0], [6.0, 6.0])
        # assert_variance(a, loss, [0.0, 0.0])


class TestGather:
    idx = dr.llvm.ad.UInt([0,0,2,3])
    def test_b(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        
        b = dr.gather(dr.llvm.ad.Float, a, self.idx)
        loss = b
        print("loss is: b")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [2, 0, 1, 1], [2, 0, 1, 1], [2, 0, 1, 1])
    
    def test_b2(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
       
        b = dr.gather(dr.llvm.ad.Float, a, self.idx)
        loss = b * 3
        print("loss is: b * 2")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        #NOT SURE whether use scatter reduce sqr(grad2) or use scatter reduce grad2
        assert_backprop(a, loss, [6, 0, 3, 3], [18, 0, 9, 9], [2, 0, 1, 1])
        # assert_backprop(a, loss, [6, 0, 3, 3], [162, 0, 81, 81], [2, 0, 1, 1])

    
    def test_bb(self):
        # when scatter sqr and 1 and leaf cannot pass, counter is not correct
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        b = dr.gather(dr.llvm.ad.Float, a, self.idx)
        loss = b + b
        print("loss is: b + b")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [4, 0, 2, 2], [4, 0, 2, 2], [4, 0, 2, 2])
        # assert_backprop(a, loss, [4, 0, 2, 2], [8, 0, 4, 4], [4, 0, 2, 2])
    

    
    def test_a2(self):
        # when scatter sqr and 1 and leaf cannot pass, grad2, counter is not correct
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        a1 = a * 2
        b = dr.gather(dr.llvm.ad.Float, a1, self.idx)
        loss = dr.sum(b)
        print("loss is: b, b = gather(a * 2)")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [4, 0, 2, 2], [8, 0, 4, 4], [2, 0, 1, 1])

    
    def test_aa(self):
        # when scatter sqr and 1 and leaf cannot pass, grad2, counter is not correct
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        a1 = a + a
        b = dr.gather(dr.llvm.ad.Float, a1, self.idx)
        loss = dr.sum(b)
        print("loss is: b, b = gather(a + a)")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [4, 0, 2, 2], [4, 0, 2, 2], [4, 0, 2, 2])

    

class TestScatterReduce:
    idx = dr.llvm.ad.UInt([0,0,2,3])
    def test_b(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)

        b = dr.llvm.ad.Float([0,0,0,0])
        
        dr.scatter_reduce(dr.ReduceOp.Add, b, a, self.idx)
        loss = b
        print("loss is: b = scatter_reduce(a)")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    
    def test_b2(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)

        b = dr.llvm.ad.Float([0,0,0,0])
        
        dr.scatter_reduce(dr.ReduceOp.Add, b, a, self.idx)
        loss = b * 2
        print("loss is: b * 2")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [2, 2, 2, 2], [4, 4, 4, 4], [1, 1, 1, 1])
    
    def test_bb(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)

        b = dr.llvm.ad.Float([0,0,0,0])
        
        dr.scatter_reduce(dr.ReduceOp.Add, b, a, self.idx)
        loss = b + b
        print("loss is: b + b")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2])
    
    def test_a2(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        a1 = a * 2
        b = dr.llvm.ad.Float([0,0,0,0])
        dr.scatter_reduce(dr.ReduceOp.Add, b, a1, self.idx)
        loss = dr.sum(b)
        print("loss is: b = scatter_reduce(a * 2)")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [2, 2, 2, 2], [4, 4, 4, 4], [1, 1, 1, 1])
    
    def test_aa(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 1.0])
        dr.enable_grad(a)
        a1 = a + a
        b = dr.llvm.ad.Float([0,0,0,0])
        dr.scatter_reduce(dr.ReduceOp.Add, b, a1, self.idx)
        loss = dr.sum(b)
        print("loss is: b = scatter_reduce(a + a)")
        print(f"b is: {b}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2])

class TestCount:
    def test_basic(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 2.0])
        dr.enable_grad(a)
        mask = dr.eq(a, 2.0)
        loss = dr.select(mask, a, 0)
        print(f"loss is: {loss}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1])

    
    def test_linear(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 2.0])
        dr.enable_grad(a)
        mask = dr.eq(a, 2.0)
        c = dr.select(mask, a, 0)
        loss = c + c
        print(f"loss is: {loss}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [0, 2, 0, 2], [0, 2, 0, 2], [0, 2, 0, 2])
    
    def test_linear2(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 2.0])
        dr.enable_grad(a)
        mask = dr.eq(a, 2.0)
        c = dr.select(mask, a, 0)
        loss = c * 2
        print(f"loss is: {loss}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [0, 2, 0, 2], [0, 4, 0, 4], [0, 1, 0, 1])
    
    def test_combine(self):
        a = dr.llvm.ad.Float([4.0, 2.0, 3.0, 2.0])
        dr.enable_grad(a)
        mask1 = dr.eq(a, 2.0)
        part1 = dr.select(mask1, a, 0)
        mask2 = dr.eq(a, 3.0)
        part2 = dr.select(mask2, a, 0)
        mask3 = dr.eq(a, 4.0)
        part3 = dr.select(mask3, a, 0)
        c = part1 + part2 + part3
        loss = c
        print(f"loss is: {loss}")
        print()
        # debug_backprop(a,loss)
        assert_backprop(a, loss, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
        


if __name__ == "__main__":
    t1d = Test1D()
    tnd = TestND()
    tg = TestGather()
    tsc = TestScatterReduce()
    tc = TestCount()
    t1d.test_basic()
    t1d.test_basic2()
    t1d.test_linear()
    t1d.test_sum()
    t1d.test_linear_array()
    t1d.test_linear_array_2()
    t1d.test_linear_array_3()
    t1d.test_product()
    t1d.test_add_product()
    t1d.test_square()

    tnd.test_basic()
    tnd.test_linear()
    tnd.test_sum()

    tg.test_b()
    tg.test_b2()
    tg.test_bb()
    tg.test_a2()
    tg.test_aa()

    tsc.test_b()
    tsc.test_b2()
    tsc.test_bb()
    tsc.test_a2()
    tsc.test_aa()


    tc.test_basic()
    tc.test_linear()
    tc.test_linear2()
    tc.test_combine()
