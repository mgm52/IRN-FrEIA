# Invertible Image Rescaling with FrEIA

This is a reimplementation of [Invertible Image Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling) using the [Freia](https://github.com/VLL-HD/FrEIA) framework

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
- Various IQA methods including DISTS https://github.com/dingkeyan93/IQA-optimization

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


### Writeup details

- Idea: draw a grid explaining the differences between compression and our rescaling task, and maybe fitting the "stamped downscaling" idea in between the two

- Idea: do some math as to the relationship between L2 Guidance Loss (how much our downscaled images deviate from bicubic) and how much information they encode to help us upscale (assuming no knowledge of the dataset from which input images are drawn). Theoretically, if the downscaling process is allowed to deviate 100% away from bicubic, it should be equivalent to a really good lossy compression
  - Flaw with this idea: there are probably multiple ways to compress stuff and may be difficult to get a single absolute rating for how much information we encode by deviating
  - Could relate to rate-distortion theory: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory
 
 - Can include explanation of JSD and how it can be substituted for our surrogate Loss_distr, which is an approximation of cross-entropy
