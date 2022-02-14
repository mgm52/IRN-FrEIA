# Invertible Image Rescaling with FrEIA

*Example results on DIV2K test set after 200 epochs (4 hours training time on an Tesla P100 GPU):*
![Example output!](/output/5e7d329f-7605-4491-82c5-b5f8ac1899aa.png "Example output")
## Links
PAPERS
- Analyzing Inverse Problems With Invertible Neural Networks https://arxiv.org/pdf/1808.04730.pdf
- Invertible image rescaling https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460120.pdf
- Invertible image rescaling w/ appendix https://arxiv.org/pdf/2005.05650.pdf
- NICE (explains normalizing flow & proposes a good coupling layer) https://arxiv.org/pdf/1410.8516.pdf
- Task-aware image downscaling (maybe relevant) https://openaccess.thecvf.com/content_ECCV_2018/papers/Heewon_Kim_Task-Aware_Image_Downscaling_ECCV_2018_paper.pdf
- Lossy Image Compression with Normalizing Flows - closely related paper https://studios.disneyresearch.com/app/uploads/2021/05/Lossy-Image-Compression-with-Normalizing-Flows.pdf

GITHUB
- IRN github https://github.com/pkuxmq/Invertible-Image-Rescaling
- FrEIA https://github.com/VLL-HD/FrEIA
Various IQA methods including DISTS https://github.com/dingkeyan93/IQA-optimization

DISS
- Dissertation structure explanation https://www.cst.cam.ac.uk/teaching/part-ii/projects/dissertation
- Assessment criteria https://www.cst.cam.ac.uk/teaching/part-ii/projects/assessment
- Projects from previous years https://www.cl.cam.ac.uk/teaching/projects/overseers/archive.html

HPC
- CSD3 application form 	https://www.hpc.cam.ac.uk/rcs-application
- CSD3 tutorial site 		https://docs.hpc.cam.ac.uk/hpc/
- HPC web portal 			https://login-web.hpc.cam.ac.uk/pun/sys/dashboard/

OTHER
- Simple normalizing flow explanation https://stats.stackexchange.com/questions/414847/difference-between-invertible-nn-and-flow-based-nn
- Scientific Linux 7 http://ftp.scientificlinux.org/linux/scientific/7x/x86_64/iso/

## Todo

### Immediate next steps

~~- Output test metrics: PSNR and FID, and log-likelihood for mnist generation~~
- ~~Train on higher resolutions ~~making use of pytorch DataLoader~~
- ~~Investigate why using ActNorm makes my pytorch implementation deviate more from FrEIA, and why FrEIA doesn't seem to quite use actnorm in allinone (?) (<-- solved, it's because parameters weren't being trained)~~
- Investigate GPU server performance issues and train for a longer period of time
- Clean up and comment on code
- Write tests
- Consider organising code into access, assess, address format (neil's "fynesse" template)

### Enhancements

- Idea: use a more visually meaningful loss function ("IQA" - image quality assessment) for reconstruction & downscaling, instead of just MSE.
  - For example, use DISTS (https://arxiv.org/pdf/2004.07728.pdf implemented https://github.com/dingkeyan93/IQA-optimization) in combination with MSE (as is done here, equation 2 page 2 https://openaccess.thecvf.com/content/CVPR2021W/CLIC/papers/Wang_Subjective_Quality_Optimized_Efficient_Image_Compression_CVPRW_2021_paper.pdf)
  - *[Param] There's been a lot of work on using a "visually meaningful loss". Most people use LPIPS or some other loss based on VGG-net. Within our group, we recently had a related project documented in a [blog](https://towardsdatascience.com/perceptual-losses-for-image-restoration-dd3c9de4113). Although I wasn't involved, I can setup a quick chat with one of the authors to see what they think.*
  - Can I equate some of the properties of JPEG in a loss function? (to make our IQA closer to human perception, and/or to make the model more resiliant to jpeg compression).
    - e.g. chromas subsampling, by which we have a higher penalty for deviating in luma than in colour
  - *[Param] Variational dequantization looks particularly appealing for compression (https://arxiv.org/abs/1902.00275). There's a nice implementation [here](https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/dequantization_variational.py).*

- Hunt down bottlenecks in training time: is it due to fetching samples?
  - *[Param] Most likely it is fetching data. With CSD3, each node has 1000 Gb RAM. You can load the entire dataset into main memory; eliminate all file-based i/o during training.*

- The first Haar block literally produces a (bicubic?) downscaled image of x. Instead of downscaling later, could I use a graph network to extract the first Haar downscaled image? Or I could define a custom inn which calls HDS then the rest of the network.
  - *[Param] You loose invertibility if you use a graph network. I think it's best to leave the Haar block as it is. It already performs the important task of frequency decomposition in an invertible way.*

- The main advantage and main constraint of normalizing flows is producing a tractably computable jacobian. The IRN model does not make use of it. Can we use more expressive coupling layers because of this?
  - *[Param] I totally agree with you, and had thought about this for another project. Unfortunately, there isn't a lot of work on expressible invertible architectures that don't require explicit likelihood. The best resource is probably Table 1 in [this paper](https://arxiv.org/pdf/1811.00995.pdf).*
  - *[Param] One of these is implemented in Freia. See if [this layer](https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html#FrEIA.modules.IResNetLayer) helps your network.*

- Try using an adversarial loss function - i.e. train a network to predict whether, given a LR and HR image, the HR is upscaled or GT.
	- Paper actually does this in the full Loss_distr_match. Says that GAN training is unstable however.
	- *[Param] See discussion on perceptual loss*

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
	- In fact, perhaps I could achieve steganography just by constraining LR to resemble a specific image??? Then we can produce a range of similar-looking images of one thing, which actually map to other stuff.
	- *[Param] You might want to check [this paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Large-Capacity_Image_Steganography_Based_on_Invertible_Neural_Networks_CVPR_2021_paper.pdf). FOund it when I was reading about a different, completely unrelated project!*

### Writeup details

- Idea: draw a grid explaining the differences between compression and our rescaling task, and maybe fitting the "stamped downscaling" idea in between the two

- Idea: do some math as to the relationship between L2 Guidance Loss (how much our downscaled images deviate from bicubic) and how much information they encode to help us upscale (assuming no knowledge of the dataset from which input images are drawn). Theoretically, if the downscaling process is allowed to deviate 100% away from bicubic, it should be equivalent to a really good lossy compression
  - Flaw with this idea: there are probably multiple ways to compress stuff and may be difficult to get a single absolute rating for how much information we encode by deviating
  - Could relate to rate-distortion theory: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory
 
 - Can include explanation of JSD and how it can be substituted for our surrogate Loss_distr, which is an approximation of cross-entropy
