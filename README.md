# Invertible Image Rescaling with FrEIA

This is a reimplementation of [Invertible Image Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling) using the [Freia](https://github.com/VLL-HD/FrEIA) framework.
This repo also contains:
- IRN research ideas in experiments.md
- Tests of the framework on MNIST
- Further experiments on IRN including compression

## Links
CORE PAPER
- Invertible image rescaling https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460120.pdf
- Invertible image rescaling w/ appendix https://arxiv.org/pdf/2005.05650.pdf
  - DIV2K 4x PSNR/SSIM: 35.07/0.9318, 4.4M params

BACKGROUND PAPERS
- Analyzing Inverse Problems With Invertible Neural Networks https://arxiv.org/pdf/1808.04730.pdf
- NICE (explains normalizing flow & proposes a good coupling layer) https://arxiv.org/pdf/1410.8516.pdf
- Task-aware image downscaling (maybe relevant) https://openaccess.thecvf.com/content_ECCV_2018/papers/Heewon_Kim_Task-Aware_Image_Downscaling_ECCV_2018_paper.pdf
- Lossy Image Compression with Normalizing Flows - closely related paper https://studios.disneyresearch.com/app/uploads/2021/05/Lossy-Image-Compression-with-Normalizing-Flows.pdf
- Various IQA methods including DISTS https://github.com/dingkeyan93/IQA-optimization

COMPETING PAPERS
- HCFlow: Hierarchical conditional flow: a unified framework for image super-resolution and image rescaling https://arxiv.org/pdf/2108.05301.pdf
  - They train on image sizes of 160x160 instead of 144...
  - They use a grad_norm clip value of 100 instead of 10...
  - They use a grad_value clip value of 5 instead of (none)...
  - They perform an additional lr step at 450000 samples...
  - They use an initial learning rate of 2.5e-4 instead of  2e-4...
  - They use a loss of (mean) (r=1, g=0.05, d=0.00001) instead of (sum) (r=1, g=16, d=1)...
  - DIV2K 4x PSNR/SSIM: 35.23/0.9346, 4.4M params
- FGRN: Approaching the Limit of Image Rescaling via Flow Guidance https://arxiv.org/pdf/2111.05133.pdf
  - Uses two non-invertible networks for `compressed<->upscaled`, one invertible network for `compressed<->downscaled`. I am slightly dubious as to how useful that really is.
  - In section 4.5, they train an IRN with z=0 instead of resampling z, and find that it achieves similar results. They conclude that this means z does not encode the information lost in downscaling. I think they might misunderstand the purpose of z. Better experiments could be to try sampling around z=0 and see if the samples achieve similar PSNR (I expect they would do if Z~N(0,1), but might not if Z=0).
  - DIV2K 4x PSNR/SSIM: 35.15/0.9322, 3.35M params
- AIDN: Scale-arbitrary Invertible Image Downscaling https://arxiv.org/pdf/2201.12576.pdf
  - Outperforms IRN on not-power-of-two image rescaling
  - DIV2K 4x PSNR/SSIM: 34.94/?, 3.8M params

CAMBRIDGE HPC
- CSD3 application form 	https://www.hpc.cam.ac.uk/rcs-application
- CSD3 tutorial site 		https://docs.hpc.cam.ac.uk/hpc/
- HPC web portal 			https://login-web.hpc.cam.ac.uk/pun/sys/dashboard/

OTHER
- Simple normalizing flow explanation https://stats.stackexchange.com/questions/414847/difference-between-invertible-nn-and-flow-based-nn
- Scientific Linux 7 http://ftp.scientificlinux.org/linux/scientific/7x/x86_64/iso/
