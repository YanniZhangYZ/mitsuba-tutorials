
import drjit as dr
import mitsuba as mi
import numpy as np




def mse(image):
    return dr.mean(dr.sqr(image - image_ref))


def run(params, key, iter_, spp_):
    grad = []
    grad2 = []
    counter = []
    iter = iter_

    # %%
    for i in range(iter):
        # it's important to set the seeds to get statistically independent samples
        image = mi.render(scene,params,seed=i,spp=spp_)      
        loss = mse(image)
        dr.backward(loss, flags = dr.ADFlag.BackPropVarianceCounter | dr.ADFlag.ClearVertices)
        g_ = dr.grad(params[key])
        g2_= dr.grad2(params[key])
        c_ = dr.counter(params[key])
        grad.append(g_)
        grad2.append(g2_)
        counter.append(c_)
        print("this is set grad from test script")
        dr.set_grad(params[key], 0)
    grad = np.array(grad, dtype=np.float64)
    grad2 = np.array(grad2, dtype=np.float64)
    counter = np.array(counter, dtype=np.float64)
    return grad, grad2, counter


if __name__ == "__main__":
    mi.set_variant('llvm_ad_rgb')
    scene = mi.load_file('/Users/yannizhang/Desktop/RGL/mitsuba3/tutorials/variance_counter_test/scenes/rectangle.xml', spp = 1, resx = 2, resy = 2, max_depth = 6)
    image_ref = scene.integrator().render(scene, scene.sensors()[0])
    params = mi.traverse(scene)
    # key = 'color_checkerboard.color1.value'
    key = 'red.reflectance.value'
    param_ref = mi.Color3f(params[key])
    params[key] = mi.Color3f(0.01, 0.2, 0.9)
    dr.enable_grad(params[key])
    params.update()

    image_init = scene.integrator().render(scene, scene.sensors()[0])
    grad, grad2, counter = run(params, key, 1, 10)
    
    print('grad: ', grad)
    print('grad2: ', grad2)
    print('counter: ', counter)