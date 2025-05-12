import os
import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import io
    import base64
    from IPython.display import HTML
    from IPython import display as ipythondisplay
except:
    print('[ utils/colab ] Warning: not importing colab dependencies')

from .serialization import mkdir
from .arrays import to_torch, to_np
from .video import save_video


def run_diffusion(model, dataset, obs, n_samples=1, device='cuda:0', **diffusion_kwargs):
  ## normalize observation for model
  obs = dataset.normalizer.normalize(obs, 'observations')

  ## add a batch dimension and repeat for multiple samples
  ## [ observation_dim ] --> [ n_samples x observation_dim ]
  obs = obs[None].repeat(n_samples, axis=0)

  ## format `conditions` input for model
  conditions = {
    0: to_torch(obs, device=device)
  }

  samples, diffusion = model.conditional_sample(conditions,
        return_diffusion=True, verbose=False, **diffusion_kwargs)

  ## [ n_samples x (n_diffusion_steps + 1) x horizon x (action_dim + observation_dim)]
  diffusion = to_np(diffusion)

  ## extract observations
  ## [ n_samples x (n_diffusion_steps + 1) x horizon x observation_dim ]
  normed_observations = diffusion[:, :, :, dataset.action_dim:]

  ## unnormalize observation samples from model
  observations = dataset.normalizer.unnormalize(normed_observations, 'observations')

  ## [ (n_diffusion_steps + 1) x n_samples x horizon x observation_dim ]
  observations = einops.rearrange(observations,
                                  'batch steps horizon dim -> steps batch horizon dim')

  return observations


def show_diffusion(renderer, observations, n_repeat=100, substep=1, filename='diffusion.mp4', savebase='/content/videos'):
    '''
        observations : [ n_diffusion_steps x batch_size x horizon x observation_dim ]
    '''
    savebase = os.path.join(os.getcwd(), "vids")
    os.makedirs(savebase, exist_ok=True)
    savepath = os.path.join(savebase, filename)

    subsampled = observations[::substep]

    images = []
    for t in tqdm(range(len(subsampled))):
        observation = subsampled[t]  # shape: [batch_size x horizon x obs_dim]

        # renderer.composite() should now return an image, not save it
        img = renderer.composite(None, observation)
        if img is None:
            raise ValueError(f"renderer.composite returned None at timestep {t}")
        images.append(img)

    images = np.array(images)

    # Pause at the last frame
    images = np.concatenate([
        images,
        images[-1:].repeat(n_repeat, axis=0)
    ], axis=0)

    # Convert to RGB if needed and ensure dtype is uint8
    if images.ndim == 4 and images.shape[-1] == 4:  # RGBA → RGB
        images = images[..., :3]
    elif images.ndim == 3:  # Grayscale → RGB
        images = np.stack([images] * 3, axis=-1)

    images = images.astype(np.uint8)

    save_video(savepath, images)
    show_video(savepath)


def show_sample(renderer, observations, filename='sample.mp4', savebase='/logs/'):
    '''
        observations : [ batch_size x horizon x observation_dim ]
    '''

    savebase = os.path.join(os.getcwd(), "vids")
    os.makedirs(savebase, exist_ok=True)
    savepath = os.path.join(savebase, filename)

    images = []
    for rollout in observations:
        ## [ horizon x height x width x channels ]
        img = renderer.renders(rollout) # fra _renders
        images.append(img)

    ## [ horizon x height x (batch_size * width) x channels ]
    images = np.concatenate(images, axis=2)
    
    save_video(savepath, images)
    show_video(savepath, height=200)


def show_samples(renderer, observations_l, figsize=12):
    '''
      observations_l : [ [ n_diffusion_steps x batch_size x horizon x observation_dim ], ... ]
    '''

    images = []
    for observations in observations_l:
      path = observations[-1]
      img = renderer.composite(None, path)
      images.append(img)
    images = np.concatenate(images, axis=0)

    plt.imshow(images)
    plt.axis('off')
    plt.gcf().set_size_inches(figsize, figsize)


def show_video(path, height=400):
  video = io.open(path, 'r+b').read()
  encoded = base64.b64encode(video)
  ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
              loop controls style="height: {0}px;">
              <source src="data:video/mp4;base64,{1}" type="video/mp4" />
           </video>'''.format(height, encoded.decode('ascii'))))
