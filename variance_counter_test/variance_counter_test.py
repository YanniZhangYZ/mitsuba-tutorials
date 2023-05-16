import drjit as dr
import mitsuba as mi

def mse(image, image_ref):
    return dr.mean(dr.sqr(image - image_ref))

def test():

    mi.set_variant('llvm_ad_rgb')
    scene = mi.load_file('/Users/yannizhang/Desktop/RGL/mitsuba3/tutorials/scenes/cbox.xml', res=128, integrator='prb')
    image_ref = mi.render(scene, spp=512)

    params = mi.traverse(scene)

    key = 'red.reflectance.value'

    # Save the original value
    param_ref = mi.Color3f(params[key])

    # Set another color value and update the scene
    params[key] = mi.Color3f(0.01, 0.2, 0.9)
    params.update()

    image_init = mi.render(scene, spp=128)

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    iteration_count = 1

    errors = []
    for it in range(iteration_count):
        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(scene, params, spp=4)
        
        # Evaluate the objective function from the current rendered image
        loss = mse(image, image_ref)

        # Backpropagate through the rendering process
        dr.backward(loss,flags = dr.ADFlag.BackpropVarianceCounter | dr.ADFlag.ClearVertices)
        print(dr.counter(opt[key][1]))
        print(dr.grad2(opt[key][1]))
        print(dr.grad(opt[key][1]))



        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        # Update the scene state to the new optimized values
        params.update(opt)
        
        # Track the difference between the current color and the true value
        err_ref = dr.sum(dr.sqr(param_ref - params[key]))
        print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
        errors.append(err_ref)
    print('\nOptimization complete.')   

if __name__ == "__main__":
    test()


