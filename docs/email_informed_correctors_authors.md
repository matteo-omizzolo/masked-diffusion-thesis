# Email Draft: informed-correctors Text8 HollowMD4

## Subject

Question about Text8 HollowMD4 checkpoint / training details for informed-correctors

## Polished Email

Dear Prof. Linderman and informed-correctors authors,

My name is Matteo Mizzolo, and I am a master’s student working with Prof. Giacomo Zanella at Bocconi University. My thesis studies finite-budget corrector timing in masked/discrete diffusion models: given a fixed predictor schedule and a limited budget of corrector calls, I am trying to understand when timing can be explained by marginal signals and when it requires interaction-aware or search-based scheduling.

Your `lindermanlab/informed-correctors` repository is especially relevant for my thesis because the Text8 HollowMD4 setup appears to expose learned conditional logits with the hollow-transformer structure, which is a much cleaner setting for studying informed corrector timing than the larger language-model backend I have been using so far.

I wanted to ask whether any of the Text8 HollowMD4 artifacts from your experiments are shareable:

- a Text8 HollowMD4 checkpoint;
- the exact Text8 training config used for the reported results;
- any vocabulary/preprocessing files or instructions, especially the word-vocabulary pickle used by `config.vocab_dir`;
- the sampling and evaluation commands used for Text8;
- the corrector sampler and hyperparameters used in the Text8 experiments;
- an expected loss/bpc/sample-quality target that would indicate a faithful reproduction.

If checkpoints are not available, even the exact training/evaluation command and any caveats about the Text8 data layout would be very helpful. I am also interested in whether you see any issue with using the trained HollowMD4 model for timing diagnostics where the corrector is called only at selected diffusion steps rather than at every step.

I would of course cite your paper and repository, and I would be happy to acknowledge any guidance if it helps the thesis experiments.

Thank you very much for your time.

Best regards,

Matteo Mizzolo

## Shorter Version

Dear Prof. Linderman and informed-correctors authors,

I am a master’s student working with Prof. Giacomo Zanella on finite-budget corrector timing in masked/discrete diffusion models. Your Text8 HollowMD4 implementation is very relevant because its hollow-transformer structure gives a clean learned-conditional setting for studying informed correctors.

Would you be able to share a Text8 HollowMD4 checkpoint, or otherwise the exact training config, preprocessing/vocabulary files, sampling/evaluation commands, corrector hyperparameters, and expected loss/bpc target for reproducing the Text8 result?

I am hoping to use the model for timing diagnostics where corrector calls are inserted only at selected diffusion steps. If there are caveats about using the code this way, I would be grateful to know them.

I will cite your paper/repo and acknowledge any help.

Best regards,

Matteo Mizzolo

## Follow-Up After 7-10 Days

Dear Prof. Linderman and informed-correctors authors,

I wanted to briefly follow up on my question about the Text8 HollowMD4 setup in `lindermanlab/informed-correctors`. If a checkpoint is not shareable, even a short pointer to the exact training command/config, Text8 preprocessing/vocabulary setup, and expected reproduction loss/bpc would be very useful.

Thank you again for your time.

Best regards,

Matteo Mizzolo
