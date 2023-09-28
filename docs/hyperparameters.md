# Hyperparameters
We provide details for every adjustable hyperparameter in this project.

This list might look long, but in practice very few hyperparameters need to be tuned, we just wanted to be thorough :)

Important hyperparameters:


| Hyperparameter        | Description                                                                                               | Recommended/Legal Values                                  |
|------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `ds`                   | Dataset name                                                                                              | "obj3d128", "traffic", "phyre", etc.                             |
| `root`                 | Root directory for dataset                                                                                 | Example: `/mnt/data/tal/obj3d/`                                |
| `device`               | Device to run the model on                                                                                 | "cuda", "cpu"                                           |
| `batch_size`           | Number of samples in each mini-batch                                                                       | Integer value                                       |
| `lr`                   | Learning rate                                                                                             | Floating-point value, `0.0002`                                |
| `num_epochs`           | Number of training epochs                                                                                  | Integer value                                       |
| `recon_loss_type`      | Type of reconstruction loss                                                                                | "vgg", "mse"                                     |
| `beta_kl`              | Weight for the KL divergence loss                                                                          | Floating-point value, recommended: 0.15 for "mse", 40.0 for "vgg"                                |
| `patch_size`           | Patch size for the prior keypoint proposals network                                                        | Integer value, recommended: [8 ,16, 32]                                     |
| `n_kp_enc`             | Number of posterior keypoints to be learned                                                                 | Integer value, we used: [10, 12, 25]                                       |
| `learned_feature_dim`  | Dimension of latent visual features extracted from glimpses                                                | Integer value, best: [4, 10]                                        |
| `n_kp_prior`           | Number of keypoints to filter from the set of prior keypoints                                              | Integer value, in practice, we don't filter (=number of prior patches)                                       |
| `anchor_s`             | Glimpse size defined as a ratio of image size, effectively the posterior patch size (e.g., 0.25 for image_size=128 -> glimpse_size=32)             | Floating-point value, best: 0.125 for `phyre`, 0.25 for all others                                |
| `enc_channels`         | Number of channels for the posterior CNN                                                                   | List of integer values, (32, 32, 64, 64, 128, 128)                              |
| `prior_channels`       | Number of channels for the prior CNN                                                                       | List of integer values (32, 32, 64)                             |
| `timestep_horizon`     | Number of timesteps to train DDLP on (DDLP)                                                           | Integer value, typical values: [10, 15, 20]                                       |
| `beta_dyn`             | Weight for the KL dynamics loss (DDLP)                                                                            | Floating-point value, recommended: =`beta_kl`                                |
| `pint_dim`             | Dimension of the transformer model (DDLP)                                                                       | Integer value, best: [256, 512]                                       |


Full list:

| Hyperparameter        | Description                                                                                               | Recommended/Legal Values                                  |
|------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `ds`                   | Dataset name                                                                                              | "obj3d128", "traffic", "phyre", etc.                             |
| `root`                 | Root directory for dataset                                                                                 | Example: `/mnt/data/tal/obj3d/`                                |
| `device`               | Device to run the model on                                                                                 | "cuda", "cpu"                                           |
| `batch_size`           | Number of samples in each mini-batch                                                                       | Integer value                                       |
| `lr`                   | Learning rate                                                                                             | Floating-point value, `0.0002`                                |
| `kp_activation`        | Activation function for keypoints                                                                          | **"tanh"** (use that), "sigmoid"                                      |
| `pad_mode`             | Padding mode for the CNNs                                                                                  | **"replicate"** (best), "zeros "                                  |
| `load_model`           | Flag to load pre-trained model                                                                             | true, false                                        |
| `pretrained_path`      | Path to the pre-trained model                                                                              | String or null                                      |
| `num_epochs`           | Number of training epochs                                                                                  | Integer value                                       |
| `n_kp`                 | Number of keypoints to extract from each patch                                                             | Integer value, recommended: 1                                       |
| `recon_loss_type`      | Type of reconstruction loss                                                                                | "vgg", "mse"                                     |
| `sigma`                | Prior standard deviation for the keypoints in Chamfer-KL                                                   | Floating-point value, unused (leave as is)                                |
| `beta_kl`              | Weight for the KL divergence loss                                                                          | Floating-point value, recommended: 0.15 for "mse", 40.0 for "vgg"                                |
| `beta_rec`             | Weight for the reconstruction loss                                                                          | Floating-point value, recommended: 1.0                                |
| `patch_size`           | Patch size for the prior keypoint proposals network                                                        | Integer value, recommended: [8 ,16, 32]                                     |
| `topk`                 | Top-k value for plotting keypoints (used only for keypoints ploting)                                       | Integer value, default: 10                                       |
| `n_kp_enc`             | Number of posterior keypoints to be learned                                                                 | Integer value, we used: [10, 12, 25]                                       |
| `eval_epoch_freq`      | Frequency of evaluation during training                                                                    | Integer value, default: 1                                      |
| `learned_feature_dim`  | Dimension of latent visual features extracted from glimpses                                                | Integer value, best: [4, 10]                                        |
| `bg_learned_feature_dim`  | Dimension of latent visual features extracted from the background                                        | Integer value, default: same as `learned_feature_dim`                                        |
| `n_kp_prior`           | Number of keypoints to filter from the set of prior keypoints                                              | Integer value, in practice, we don't filter (=number of prior patches)                                       |
| `weight_decay`         | Weight decay for the optimizer                                                                             | Floating-point value, default: 0.0                                |
| `kp_range`             | Range of keypoints                                                                                        | **[-1, 1]** (use that), [0, 1]                                    |
| `warmup_epoch`         | Number of warm-up epochs for DLP, where only the patch encoder/decoder is trained                          | Integer value, default: 1                                       |
| `dropout`              | Dropout rate for the CNNs                                                                                  | Floating-point value, default: 0.0                                |
| `iou_thresh`           | IoU threshold for object bounding boxes (only for plotting)                                                | Floating-point value, default: 0.2                                |
| `anchor_s`             | Glimpse size defined as a ratio of image size, effectively the posterior patch size (e.g., 0.25 for image_size=128 -> glimpse_size=32)             | Floating-point value, best: 0.125 for `phyre`, 0.25 for all others                                |
| `kl_balance`           | Balance parameter between attributes and visual appearance features for the KL divergence loss          | Floating-point value, best: 0.001                                |
| `image_size`           | Size of the input image                                                                                    | Integer value, e.g., 128                                       |
| `ch`                   | Number of channels in the input image                                                                      | Integer value (=3)                                       |
| `enc_channels`         | Number of channels for the posterior CNN                                                                   | List of integer values, (32, 32, 64, 64, 128, 128)                              |
| `prior_channels`       | Number of channels for the prior CNN                                                                       | List of integer values (32, 32, 64)                             |
| `timestep_horizon`     | Number of timesteps to train DDLP on (DDLP)                                                           | Integer value, typical values: [10, 15, 20]                                       |
| `predict_delta`        | Flag to predict the delta between consecutive frames instead of absolute coordinates (DDLP)                     | true (use that), false
| `beta_dyn`             | Weight for the KL dynamics loss (DDLP)                                                                            | Floating-point value, recommended: =`beta_kl`                                |
| `scale_std`            | Prior standard deviation for scale                                                                         | Floating-point value, recommended: [0.3, 1.0]                                |
| `offset_std`           | Prior standard deviation for offset                                                                        | Floating-point value, recommended: [0.2, 1.0]                                |
| `obj_on_alpha`         | Prior alpha (Beta distribution) for obj_on (transparency)                                                     | Floating-point value, recommended: 0.1                                |
| `obj_on_beta`          | Prior beta (Beta distribution) for obj_on (transparency)                                                       | Floating-point value, recommended: 0.1                                |
| `beta_dyn_rec`         | Weight for the dynamics reconstruction loss (DDLP)                                                              | Floating-point value, recommended: 1.0                                |
| `num_static_frames`    | Number of static frames (="burn-in frames") in the sequence that for which their KL is optimized w.r.t constant prior (DDLP) | Integer value, best: 4                                       |
| `pint_layers`          | Number of transformer layers in the dynamics module (DDLP)                                                        | Integer value, best: 6                                       |
| `pint_heads`           | Number of transformer heads in the dynamics module (DDLP)                                                         | Integer value, best: 8                                       |
| `pint_dim`             | Dimension of the transformer model (DDLP)                                                                       | Integer value, best: [256, 512]                                       |
| `run_prefix`           | Prefix for the run directory name                                                                          | String                                              |
| `animation_horizon`    | Number of frames to animate into the future ,only at inference time (DDLP)                              | Integer value, default: 50                                       |
| `eval_im_metrics`      | Flag to enable evaluation of image metrics                                                                 | true, false                                        |
| `use_resblock`         | Flag to use residual blocks in the CNNs                                                                    | true, **false** (use that)                                        |
| `scheduler_gamma`      | Learning rate scheduler gamma parameter                                                                    | Floating-point value, default: 0.95                                |
| `adam_betas`           | Beta values for the Adam optimizer                                                                         | List of floating-point values, default: (0.9, 0.999)                      |
| `adam_eps`             | Epsilon value for the Adam optimizer                                                                       | Floating-point value, default: 0.0001                                |
| `train_enc_prior`      | Flag to train the encoder prior model                                                                      | **true**, false                                        |
| `start_dyn_epoch`      | Epoch at which to start training the dynamics model (DDLP)                                                       | Integer value, default: 0                                       |
| `cond_steps`           | Number of consecutive steps to condition on in the dynamics model at infernece time (DDLP)                                         | Integer value                                       |
| `animation_fps`        | Frames per second for generating animations (DDLP)                                                              | Floating-point value, default: 0.06                                |
| `use_correlation_heatmaps` | Flag to use correlation heatmaps for tracking                                                          | **true**, false                                        |
| `enable_enc_attn`      | Flag to enable attention between patches in the particle encoder                                           | true, **false**                                        |
| `filtering_heuristic`  | Filtering heuristic to filter posterior anchors from prior keypoints                                                             | "distance", **"variance"**, "random", "none"                    |
| `use_tracking`         | Flag to enable object tracking                                                                            | true (DDLP), false (DLP)                                        |

