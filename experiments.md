### Enhancements
- Multiscale architecture
  - The architecture described in the paper downsamples images but retains latents through the entire forward pass. A drawback of this is that after each downsample, the intermediate features have same dimensionality as the original HR image (it has just been reshaped into more channels).
  - Most papers (including RealNVP, Glow, SRFlow) show that reducing dimensions after downsampling imrpoves generative sample quality quite a bit. The advantage is that you can do more focussed processing with possibly smaller subnets as you go deeper. The channels that are split off are  all added to the latent directly.
  - I don't know why IRN didn't do this. For image rescaling, using a multiscale architecture is a no-brainer. Here's how you would do this in Freia: [code](https://github.com/VLL-HD/conditional_INNs/blob/master/colorization_minimal_example/model.py) and [paper](https://arxiv.org/pdf/1907.02392.pdf) (see fig. 8) 
- Idea: use a more visually meaningful loss function ("IQA" - image quality assessment) for reconstruction & downscaling, instead of just MSE.
  - For example, use DISTS (https://arxiv.org/pdf/2004.07728.pdf implemented https://github.com/dingkeyan93/IQA-optimization) in combination with MSE (as is done here, equation 2 page 2 https://openaccess.thecvf.com/content/CVPR2021W/CLIC/papers/Wang_Subjective_Quality_Optimized_Efficient_Image_Compression_CVPRW_2021_paper.pdf)
  - *[Param] There's been a lot of work on using a "visually meaningful loss". Most people use LPIPS or some other loss based on VGG-net. Within our group, we recently had a related project documented in a [blog](https://towardsdatascience.com/perceptual-losses-for-image-restoration-dd3c9de4113). Although I wasn't involved, I can setup a quick chat with one of the authors to see what they think.*
  - *[Param] In addition, you could look at better designed functions for your loss. L1 works pretty well (see (this)[]), but there are other functions that could work even better. I have seen a [smoothed L1](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html), or [Huber loss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) in a few papers and [Charbonnier penalty](https://github.com/twtygqyy/pytorch-SRDenseNet/blob/a3185aa9838d1746a6c133caa7b57aaad1e40fd0/srdensenet.py#L134) in a few others. If you want to dig deeper, I recommend reading [this](https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf) excellent paper.*
  - Can I equate some of the properties of JPEG in a loss function? (to make our IQA closer to human perception, and/or to make the model more resiliant to jpeg compression).
    - e.g. chromas subsampling, by which we have a higher penalty for deviating in luma than in colour
  - *[Param] Variational dequantization looks particularly appealing for compression (https://arxiv.org/abs/1902.00275). There's a nice implementation [here](https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/dequantization_variational.py).*

- Hunt down bottlenecks in training time: is it due to fetching samples?
  - *[Param] Most likely it is fetching data. With CSD3, each node has 1000 Gb RAM. You can load the entire dataset into main memory; eliminate all file-based i/o during training.*

- The first Haar block literally produces a (bicubic?) downscaled image of x. Instead of downscaling later, could I use a graph network to extract the first Haar downscaled image? Or I could define a custom inn which calls HDS then the rest of the network.

- The main advantage and main constraint of normalizing flows is producing a tractably computable jacobian. The IRN model does not make use of it. Can we use more expressive coupling layers because of this?
  - *[Param] I totally agree with you, and had thought about this for another project. Unfortunately, there isn't a lot of work on expressible invertible architectures that don't require explicit likelihood. The best resource is probably Table 1 in [this paper](https://arxiv.org/pdf/1811.00995.pdf).*
  - *[Param] One of these is implemented in Freia. See if [this layer](https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html#FrEIA.modules.IResNetLayer) helps your network. This makes sense because ResNets have been tremendously helpful for similar reconstruction tasks in the past few years.*

- Try using an adversarial loss function - i.e. train a network to predict whether, given a LR and HR image, the HR is upscaled or GT.
	- Paper actually does this in the full Loss_distr_match. Says that GAN training is unstable however.
	- *[Param] I wouldn't bother using adversarial (GAN training is frustratingly unstable) or MMD loss (you most likely won't see benifits for high dimensional data).*

- Explore tips and tricks FrEIA page https://vll-hd.github.io/FrEIA/_build/html/tutorial/tips_tricks_faq.html
### Open experiments / questions

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
    - *[Param] You try to absorb loss_distr_surrogate within loss_recon. If you know that you want z to be normal, why try to enforce this in a roundabout manner though? Implicit tricks often make sense only if you can't achieve a task explicitly!*
- Explore the purpose of the *probabilistic* disentanglement. Would it not work equally well (or better?) to just take the latent distribution to be z=0, rather than adding in an element of random chance?
	- I suppose one advantage is we can visualize the realm of possible upscalings. Paper doesn't explore this very much.
	- It means we can establish a bijection between x and z, and so by optimizing z we optimize x?
  - *[Param] If z was constant, the mapping can't be bijective since there is a reduction in dimensions. The point of the known latent is to capture stochasticity in a controllable manner.*
  - **UPDATE** [Max]: [This paper (section 4.5)](https://arxiv.org/pdf/2111.05133.pdf) actually did try taking the latent distribution to be z=0, and did find that it achieves similar/better(!) results like I hypothesized. What I meant by "take the latent distribution to be z=0" was "train the network by resampling from z=0 instead of from z~N(0, 1)". The mapping is still bijective, it's just that the z produced by the network approaches 0 as we train.

- Greater iteration on loss parameters - try dropping L_guide, etc.
	- Worth noting that the inclusion of other parameters slows the speed at which loss_distr is improved.

- I should show how the downscaled image changes as loss_guide decreases. Might be able to comment on what kind of tricks it uses to compress data.

- Produce graphs of loss vs batch_size, learning_rate, learning_rate adjustment, time, etc.
	- Preliminary result: batch_size=5 reaches ~~learning_rate~~ loss of 260 twice as batch_size=1 over time on my 3080 GPU

- What does repeated application of the network look like?

- How well does the network perform when given images downscaled by other means (i.e. test IRN on super-resolution tasks)?

- Can I perform some sort of analysis on which nodes or weights are most useful within the network? 
	- I expect there are some established methods of doing this.
	- Idea: report histogram of weight values, histogram of sum of weight values leaving each node.
	- Idea: test network's performance after cutting out quantities of low-use nodes.
	- Would be helpful to also perform this analysis on a rival network architecture. Perhaps I could find a model on which some analysis like this has been done already.

### New functionality

- Optimize for very high resolutions, e.g. on 4k images. The "wavelet flow: fast training of high resolution normalizing flows" paper makes the case that their architecture can work on high res data better. Try to work this into my results and get an improvement on old architecture!

- Could I remove loss_guide to turn this into a compression technique?
	- Probably yes, but it might not be very useful given that compression has been attempted before by networks not limited by compressing to an image
	- Update: this was addressed [in a follow-up paper](https://arxiv.org/pdf/2006.11999.pdf) by the authors of IRN! They use an invertible network they call ILC. Uses Haar downsampling again, and its coupling layers are affine-affine rather than additive-affine. I should read into this.

- Could I remove loss_guide and use non-enhanced affine coupling to turn this into a general upscaling technique?
	- (context: I think loss_guide is only necessary with enhanced coupling, not affine coupling. Affine coupling doesnt change x1, but enhanced does.)
	- I think the answer is yes.
  - More generally, it's very interesting to compare the different coupling layers. With loss_guide and enhanced coupling - i.e. with the ability to change the downscaled image - we get radically great upscaling results.

- Idea: allow a section of the downscaled image - e.g. a 1-pixel border - to not conform to the original. Let the model use this 1-pixel border as a kind of recipe describing how to re-upscale the image more effectively. Call it a "rescaling stamp".
	- Do some maths as to the max amount of information this can encode assuming no knowledge of the dataset from which images are drawn?
	- Explore other styles of stamp, e.g. 1 pixel in each corner

- Idea: make the downscaling/upscaling technique durable to other forms of compression? Could add an additional loss term: reconstruction of x from JPEG-compressed y.

- Idea: steganography! Given LR and HR, return HR_steg which resembles HR but hides LR, and return latent vector z. (LR, HR) <-> (z, HR_steg)
	- In fact, perhaps I could achieve steganography just by constraining LR to resemble a specific image??? Then we can produce a range of similar-looking images of one thing, which actually map to other stuff.
	- *[Param] You might want to check [this paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Large-Capacity_Image_Steganography_Based_on_Invertible_Neural_Networks_CVPR_2021_paper.pdf). FOund it when I was reading about a different, completely unrelated project!*

- Idea: recurrent video prediction.
	- What if we set up an INN with the following input<->output: [frame n+1][frame n]<->[frame n+2][latent vector].
	- Or, perhaps a format that's more fitting for coupling layers: [frame n+1][diff(frame n, frame n+1)]<->[frame n+2][latent vector]. This way, by treating [frame n+1] as our h1 to coupling layers, the network with default values will produce something vaguely like: [frame n+1][frame n]<->[distorted frame n+1][latent vector].

- Idea: frame interpolation.
	- What if we set up an INN with the following input<->output: [frame n+2][diff(frame n, frame n+2)]<->[frame n+1][latent vector].

- Idea: video rescaling.
	- This was an idea Cengiz actually suggested during my progress presentation.
	- We would need some mechanism to ensure temporal stability across frames.
		- Simplest way: replace our loss function with a GAN that discriminates between GT pairs of frames and irn-upscaled pairs, as well as between bicubic-downscaled pairs and irn-downscaled pairs. In fact, we may be able to get away without even changing the model architecture at all: just expect it to learn to up/downscale in a more stable manner given the loss function.
		- More creative way: instead of comparing two frames using a GAN, take any image and synthesize a "next frame" by adding random noise, translation, rescaling, rotation, etc. For our loss function, check that `irn_downupscaled(gt_img)->irn_downupscaled(distort(gt_img))` is the same transformation as `gt_img->distort(gt_img)`. In other words, check that `irn_downupscaled(distort(gt_img)) == distort(irn_downupscaled(gt_img))`. To make it more accurate-to-life I could perhaps replace `distort()` with an optical-flow based motion prediction from real frames or some such.
			- Update: it turns out this idea has ben explored by Rafal Mantiuk! This 2019 paper describes a couple of different transformations similar to what I described... https://arxiv.org/pdf/1902.10424.pdf
