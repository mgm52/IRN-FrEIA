# Invertible Image Rescaling with FrEIA

*Example results on an mnist8 test set after 250 epochs (1 minute training time on an RTX 3080 GPU):*
![Example output!](/output/out_1643816086_0.png "Example output")

## Todo

### Immediate next steps

- Output test metrics: PSNR and FID, and log-likelihood for mnist generation
- Train on higher resolutions, making use of pytorch DataLoader
- Clean up and comment on code
- Organise code into access, assess, address format (neil's "fynesse" template)

### Enhancements

- Idea: use a more visually meaningful loss function ("IQA" - image quality assessment) for reconstruction & downscaling, instead of just MSE.
  - For example, use DISTS (https://arxiv.org/pdf/2004.07728.pdf implemented https://github.com/dingkeyan93/IQA-optimization) in combination with MSE (as is done here, equation 2 page 2 https://openaccess.thecvf.com/content/CVPR2021W/CLIC/papers/Wang_Subjective_Quality_Optimized_Efficient_Image_Compression_CVPRW_2021_paper.pdf)
  - Can I equate some of the properties of JPEG in a loss function? (to make our IQA closer to human perception, and/or to make the model more resiliant to jpeg compression).
    - e.g. chromas subsampling, by which we have a higher penalty for deviating in luma than in colour

- Hunt down bottlenecks in training time: is it due to fetching samples?

- The first Haar block literally produces a (bicubic?) downscaled image of x. Instead of downscaling later, could I use a graph network to extract the first Haar downscaled image? Or I could define a custom inn which calls HDS then the rest of the network.

- The main advantage and main constraint of normalizing flows is producing a tractably computable jacobian. The IRN model does not make use of it. Can we use more expressive coupling layers because of this?

- Try using an adversarial loss function - i.e. train a network to predict whether, given a LR and HR image, the HR is upscaled or GT.
	- Paper actually does this in the full Loss_distr_match. Says that GAN training is unstable however.

- Explore tips and tricks FrEIA page https://vll-hd.github.io/FrEIA/_build/html/tutorial/tips_tricks_faq.html

### Open experiments

- Is the loss_distribution_match term really necessary? Seems to me that it just bolsters the reconstruction loss metric. How do results compare if I drop loss_dist_match and put more emphasis on recon?
  - Paper tried dropping L_distr on page 13. However, it generally only found marginal improvements keeping L_distr, without much discussion.
  - Could consider writing some proof about how using Loss_Distr surrogate to enforce z~N(0,1) is equivalent to using Loss_Recon to enforce that f_inv(f(x).y, z_from_N(0,1))==x
    - Maybe something like:
      ```
      we're enforcing f_inv(f(x).y, z_from_N(0,1))==x
      so f(f_inv(f(x).y, z_from_N(0,1)))==f(x)
      so (f(x).y, z_from_N(0,1))==f(x)
      so (f(x).y, z_from_N(0,1))==(f(x).y, f(x).z)
      so z_from_N(0,1)==f(x).z
      ```
    - Maybe in THIS case (the case that loss_distr_surrogate is equivalent to loss_recon) the purpose of the distr loss is so that the loss lambdas we choose are still relevant in pre-training as they are in training?

- Explore the purpose of the *probabilistic* disentanglement. Would it not work equally well (or better?) to just take the latent distribution to be z=0, rather than adding in an element of random chance?
	- I suppose one advantage is we can visualize the realm of possible upscalings. Paper doesn't explore this very much.
	- It means we can establish a bijection between x and z, and so by optimizing z we optimize x?

- Greater iteration on loss parameters - try dropping L_guide, etc.
	- Worth noting that the inclusion of other parameters slows the speed at which loss_distr is improved.

- I should show how the downscaled image changes as loss_guide decreases. Might be able to comment on what kind of tricks it uses to compress data.

- Produce graphs of loss vs batch_size, learning_rate, learning_rate adjustment, etc.
	- Preliminary result: batch_size=5 reaches learning_rate of 260 twice as fast as batch_size=1 on my 3080

### New functionality

- Optimize for very high resolutions, e.g. on 4k images. The "wavelet flow: fast training of high resolution normalizing flows" paper makes the case that their architecture can work on high res data better. Try to work this into my results and get an improvement on old architecture!

- Could I remove loss_guide to turn this into a compression technique?
	- Probably yes, but it might not be very useful given that compression has been attempted before by networks not limited by compressing to an image

- Could I remove loss_guide and use non-enhanced affine coupling to turn this into a general upscaling technique?
	- (context: I think loss_guide is only necessary with enhanced coupling, not affine coupling. Affine coupling doesnt change x1, but enhanced does.)
	- I think the answer is yes.
  - More generally, it's very interesting to compare the different coupling layers. With loss_guide and enhanced coupling - i.e. with the ability to change the downscaled image - we get radically great upscaling results.

- Idea: allow a section of the downscaled image - e.g. a 1-pixel border - to not conform to the original. Let the model use this 1-pixel border as a kind of recipe describing how to re-upscale the image more effectively. Call it a "rescaling stamp".
	- Do some maths as to the max amount of information this can encode assuming no knowledge of the dataset from which images are drawn?
	- Explore other styles of stamp, e.g. 1 pixel in each corner

- Idea: make the downscaling/upscaling technique durable to other forms of compression? Could add an additional loss term: reconstruction of x from JPEG-compressed y.

- Idea: steganography! Given LR and HR, return HR_steg which resembles HR but hides LR, and return latent vector z. (LR, HR) <-> (z, HR_steg)

### Writeup details

- Idea: draw a grid explaining the differences between compression and our rescaling task, and maybe fitting the "stamped downscaling" idea in between the two

- Idea: do some math as to the relationship between L2 Guidance Loss (how much our downscaled images deviate from bicubic) and how much information they encode to help us upscale (assuming no knowledge of the dataset from which input images are drawn). Theoretically, if the downscaling process is allowed to deviate 100% away from bicubic, it should be equivalent to a really good lossy compression
  - Flaw with this idea: there are probably multiple ways to compress stuff and may be difficult to get a single absolute rating for how much information we encode by deviating
  - Could relate to rate-distortion theory: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory
 
 - Can include explanation of JSD and how it can be substituted for our surrogate Loss_distr, which is an approximation of cross-entropy
