# (D)DLP Graphical User Interface (GUI) Usage Instructions


**IMPORTANT NOTE**: DDLP only works with small latent modifications, and cannot handle modifications that result in out-of-distribution examples .

General usage:
1. Choosing a pre-trained model: the GUI looks for models inside the `checkpoints`" directory. The GUI supports 3 types
of models: [`dlp`, `ddlp`, `diffuse-ddlp`], and the pre-trained paths should be organized as follows:
`checkpoints/{model-type}-{ds}/[hparams.json, {ds}_{model-type}.pth]` for `dlp`/`ddlp`
and `checkpoints/diffuse-ddlp-{ds}/[ddlp_hparams.json,diffusion_hparams.json, {ds}_ddlp.pth, latent_stats.pth, /saves/model.pth]`
for `diffuse-ddlp`. For example: `checkpoints/ddlp-obj3d128/[hparams.json, obj3d128_ddlp.pth]`.


```
checkpoints
├── ddlp-obj3d128
│   ├── obj3d128_ddlp.pth
│   ├── hparams.json
├── dlp-traffic
├── diffuse-ddlp-obj3d128
│   ├── diffusion_hparams.json
│   ├── ddlp_hparams.json
│   ├── latent_stats.pth
│   ├── saves
│   │   ├── model.pth
└── ...
```

3. Choosing/generating an example: For `dlp/ddlp` the GUI looks for examples for a dataset in the `assets` directory,
where each example is a directory with an integer number as its name.
Under each example directory there should be images, where for `ddlp` at least 4 consecutive are required,
numbered by their order. For example, `assets/obj3d128/428/[1.png, 2.png, ...]`. 
For `diffuse-ddlp`, press the `Generate button` to generate a sequence of 4 (latent) frames.

4. Choosing a device: the GUI can run everything on the CPU, but if CUDA is available,
you can switch to a GPU to perform computations.

5. Animating latent transitions: if the `animate` checkbox is marked, the GUI will animate the latent interpolation 
between modifications after the `Update` button is pressed (naturally, this is slower).

6. Hiding particles: you can temporarily hide the particles to view the current image 
by marking the `hide particles` checkbox. Removing the check will restore the particles.

7. Latent modifications: the GUI supports the following modifications:
* Moving particles by dragging. Use the selection tool to select multiple particles at once and then drag
them all together. This is useful when objects are assigned multiple particles.
* Changing scale/transparency: once a particle is pressed on, a modifications menu will open.
You can change the scale and transparency for multiple particles at once by first pressing a particle,
and then using the selection tool to select multiple particles and all changes applied to the pressed particle
will be applied to all of them.
* Changing visual appearance: when an example is selected/generated, the visual features of all particles are saved
in a dictionary where the key is the particle number. You can choose between these available features.
Similarly, you can change the features of multiple features at once by first pressing on a particle and 
then use the selection tool to pick all the particles that will be changed.

8. Video prediction: when using a `ddlp`-based model, you can unroll the latent particles and generate a video in
a new window by pressing the `Play` button.
Note for DDLP, to make things simpler, we only allow changes to particles at t=0 and t=3 and interpolate all
particles in-between.

Note that DDLP is quite sensitive to out-of-distribution modifications.
