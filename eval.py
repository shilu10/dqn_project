import random 
import imageio
from policy import greedy_policy

def eval_model(env, model, out_directory, fps=20):
    images = []  
    done = False
    state = env.reset(seed=random.randint(0,500))
    img = env.render()
    images.append(img)
    while not done:
        action = greedy_policy(state, model, ACTION_SPACE)
        state, reward, done, info, _ = env.step(action) 
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)