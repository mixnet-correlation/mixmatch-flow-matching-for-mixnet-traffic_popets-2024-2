# "MixMatch: Flow Matching for Mixnet Traffic", PoPETs 2024.2

Main repository for our PoPETs 2024.2 article "MixMatch: Flow Matching for Mixnet Traffic".

Authors: [Lennart Oldenburg](https://lennartoldenburg.de/), [Marc Juarez](https://mjuarezm.github.io/), [Enrique Argones Rúa](https://www.esat.kuleuven.be/cosic/people/enrique-argones-rua/), [Claudia Diaz](https://homes.esat.kuleuven.be/~cdiaz/)


## Abstract

> Mixnets provide communication anonymity against network adversaries by routing packets independently via multiple hops, delaying them artificially at each hop, and introducing cover traffic. We show that these features (particularly the use of cover traffic) significantly diminish the effectiveness of state-of-the-art flow correlation techniques developed to link the two ends of a Tor connection. In this work, we propose novel methods to determine whether a set of endpoints exchanges packets via a mixnet and demonstrate their effectiveness by applying them to the Nym mixnet. We consider Nym in both an idealized lab setup and the official live network, and propose and compare three classifiers to conduct flow matching on it. Our statistical classifier tests whether egress packet timestamps are consistent with ingress timestamps and the (known) routing delay characteristic of the mixnet. In contrast, our two deep learning (DL) classifiers learn to distinguish matched from unmatched flow pairs from collected datasets directly, rather than relying on priors that describe the delay distribution. All three classifiers use our flow merging technique, which enables testing a match for sets of communicating endpoints of any cardinality. Considering a use case where two observed endpoints communicate exclusively to exchange a file through Nym, we find that flow matching is fast and accurate in the idealized lab setup. If flow pairs are aligned using all network observations in a download, we achieve a TPR of circa 0.6 (DL) and 0.47 (statistical) at an FPR of 10^−2 after only processing 100 observations. We evaluate classifier performance under key variations of this setup: the absence of loop cover traffic, an increased or decreased average per-mix delay, larger communicating sets (three endpoints) with faster responders, and the presence of realistic network effects (live network). The classifiers' matching performance diminishes on the live network where packet losses and variable propagation delays exist, reducing DL TPR to circa 0.26 and statistical TPR to circa 0.28 at an FPR of 10^−2. Informed by the insights of our analyses, we outline countermeasures that can be deployed in mixnets such as Nym to mitigate flow matching threats.


## Classifiers

Before we provide [step-by-step instructions](#steps-to-reproduce-results) for how to use the individual components that make up our paper's artifact, we link all repositories that we use in below instructions.

First, the three classifiers we present in our paper are:
* our **statistical classifier**: [`mixmatch_statistical_classifier`](https://github.com/mixnet-correlation/mixmatch_statistical_classifier),
* our **drift classifier** (using deep learning methods): [`mixmatch_drift_classifier`](https://github.com/mixnet-correlation/mixmatch_drift_classifier),
* and our **shape classifier** (using deep learning methods): [`mixmatch_shape_classifier`](https://github.com/mixnet-correlation/mixmatch_shape_classifier).


## Datasets

Below, we list each used dataset by its name in the paper.

* **`baseline`** (sometimes referred to as: `exp01`):
  * Description: Baseline Nym at version `nym-binaries-1.0.2` with patches to collect Sphinx flow metadata, but no changes to Nym's network or timing behavior.
  * Set of Nym patches: [`data-collection_1_experiments/exp01_nym-binaries-1.0.2_static-http-download`](https://github.com/mixnet-correlation/data-collection_1_experiments/tree/main/exp01_nym-binaries-1.0.2_static-http-download)
  * Input format for drift and shape classifiers (circa 2.8 GB): [`dataset_baseline_nym-binaries-1.0.2_deeplearning`](https://github.com/mixnet-correlation/dataset_baseline_nym-binaries-1.0.2_deeplearning)
  * Input format for statistical classifier (circa 5.6 GB): [`dataset_baseline_nym-binaries-1.0.2_statistical`](https://github.com/mixnet-correlation/dataset_baseline_nym-binaries-1.0.2_statistical)

* **`no-cover`** (sometimes referred to as: `exp02`):
  * Description: Disabled foreground ("gap-filling") and background cover traffic on endpoints.
  * Set of Nym patches: [`data-collection_1_experiments/exp02_nym-binaries-1.0.2_static-http-download_no-client-cover-traffic`](https://github.com/mixnet-correlation/data-collection_1_experiments/tree/main/exp02_nym-binaries-1.0.2_static-http-download_no-client-cover-traffic)
  * Input format for drift and shape classifiers (circa 1.2 GB): [`dataset_no-cover_nym-binaries-1.0.2_deeplearning`](https://github.com/mixnet-correlation/dataset_no-cover_nym-binaries-1.0.2_deeplearning)
  * Input format for statistical classifier (circa 2.5 GB): [`dataset_no-cover_nym-binaries-1.0.2_statistical`](https://github.com/mixnet-correlation/dataset_no-cover_nym-binaries-1.0.2_statistical)

* **`low-delay`** (sometimes referred to as: `exp05`):
  * Description: Endpoints send messages with lower average per-mix delay (`µ = 20ms` instead of `µ = 50ms`).
  * Set of Nym patches: [`data-collection_1_experiments/exp05_nym-binaries-1.0.2_static-http-download_shorter-mix-delay`](https://github.com/mixnet-correlation/data-collection_1_experiments/tree/main/exp05_nym-binaries-1.0.2_static-http-download_shorter-mix-delay)
  * Input format for drift and shape classifiers (circa 2.7 GB): [`dataset_low-delay_nym-binaries-1.0.2_deeplearning`](https://github.com/mixnet-correlation/dataset_low-delay_nym-binaries-1.0.2_deeplearning)
  * Input format for statistical classifier (circa 5.7 GB): [`dataset_low-delay_nym-binaries-1.0.2_statistical`](https://github.com/mixnet-correlation/dataset_low-delay_nym-binaries-1.0.2_statistical)

* **`high-delay`** (sometimes referred to as: `exp06`):
  * Description: Endpoints send messages with higher average per-mix delay (`µ = 200ms` instead of `µ = 50ms`).
  * Set of Nym patches: [`data-collection_1_experiments/exp06_nym-binaries-1.0.2_static-http-download_longer-mix-delay`](https://github.com/mixnet-correlation/data-collection_1_experiments/tree/main/exp06_nym-binaries-1.0.2_static-http-download_longer-mix-delay)
  * Input format for drift and shape classifiers (circa 3.1 GB): [`dataset_high-delay_nym-binaries-1.0.2_deeplearning`](https://github.com/mixnet-correlation/dataset_high-delay_nym-binaries-1.0.2_deeplearning)
  * Input format for statistical classifier (circa 6.3 GB): [`dataset_high-delay_nym-binaries-1.0.2_statistical`](https://github.com/mixnet-correlation/dataset_high-delay_nym-binaries-1.0.2_statistical)

* **`two-to-one`:**
  * Description: Based on `baseline`, a logical responder consisting of two original responders exchanges packets with two initiators.
  * We construct this dataset ad-hoc during the execution of the deep learning and statistical classifiers dedicated to this configuration.

* **`live-nym`** (sometimes referred to as: `exp08`):
  * Description: Live-Nym-network experiment with minimal behavioral modifications and a much more recent Nym version (`nym-binaries-v1.1.13`).
  * Set of Nym patches: [`data-collection_1_experiments/exp08_nym-binaries-v1.1.13_static-http-download`](https://github.com/mixnet-correlation/data-collection_1_experiments/tree/main/exp08_nym-binaries-v1.1.13_static-http-download)
  * Input format for drift and shape classifiers (circa 3.0 GB): [`dataset_live-nym_nym-binaries-v1.1.13_deeplearning`](https://github.com/mixnet-correlation/dataset_live-nym_nym-binaries-v1.1.13_deeplearning)
  * Input format for statistical classifier (circa 6.0 GB): [`dataset_live-nym_nym-binaries-v1.1.13_statistical`](https://github.com/mixnet-correlation/dataset_live-nym_nym-binaries-v1.1.13_statistical)


## Dataset Collection

Finally, we collected above datasets using the following three repositories:
* the **experiment patch sets** that modify Nym in the way we need it for a respective experiment: [`data-collection_1_experiments`](https://github.com/mixnet-correlation/data-collection_1_experiments),
* the orchestrator to collect datasets in the **isolated setup**: [`data-collection_2_isolated-setup`](https://github.com/mixnet-correlation/data-collection_2_isolated-setup),
* the orchestrator to collect datasets in the **live-network setup**: [`data-collection_3_live-network-setup`](https://github.com/mixnet-correlation/data-collection_3_live-network-setup).


## Steps to Reproduce Results

### Hardware Considerations

Please find further details on hardware and time requirements for our analyses in Appendix G of our paper.

For running our flow matching attack with the statistical classifier, we recommend to use an enterprise-grade machine with many CPU cores and sufficient RAM. As an impression, we evaluated our statistical classifier on dataset `live-nym` on a machine with 48 CPUs and 192 GB RAM and this process took roughly three days.

For running our flow matching attack with the deep learning classifiers drift and shape, we recommend to use an enterprise-grade machine equipped with a powerful GPU. As an impression, training our drift and shape classifiers for 100 epochs on dataset `baseline` on a machine with 64 CPUs, 192 GB RAM, and an NVIDIA GeForce RTX 3080 graphics card with 10 GB video memory took circa five days for the shape classifier and circa two days for the drift classifier.

In case you want to collect raw data yourself (instead of relying on above-listed already prepared datasets), a decently powerful desktop-/server-grade machine is required for data collection in the Isolated Setup, while two moderately powerful cloud machines (at least one of them with a public IP address) are needed for data collection in the Live-Network Setup.

### Setting Up

Execute the following steps as user `root` on a Ubuntu 22.04 machine `ubuntu2204` equipped with the above mentioned hardware capabilities (CPU cores, RAM size, GPU available) and at least 100 GB of free disk space. Mind that the scripts below will install some Ubuntu packages as well as Miniconda and a Miniconda-based Python environment with the packages we need. If you don't want this to happen to your current machine, please make sure to run this in a virtual machine or ephemeral cloud instance.
```bash
root@ubuntu2204 $   mkdir -p ~/mixmatch
root@ubuntu2204 $   cd ~/mixmatch
root@ubuntu2204 $   git clone https://github.com/mixnet-correlation/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2.git
root@ubuntu2204 $   cd ~/mixmatch/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2
root@ubuntu2204 $   ./1_setup_1_system.sh
root@ubuntu2204 $   systemctl reboot
```

Wait for `ubuntu2204` machine to boot back up, then continue with:
```bash
root@ubuntu2204 $   cd ~/mixmatch/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2
root@ubuntu2204 $   ./1_setup_2_conda.sh
root@ubuntu2204 $   ~/miniconda3/bin/conda init bash   # Activate conda, modify if you use a different shell
root@ubuntu2204 $   exit
```

Log back into the instance `ubuntu2204` and run:
```bash
root@ubuntu2204(base) $   cd ~/mixmatch/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2
root@ubuntu2204(base) $   ./1_setup_3_conda-packages.sh
... This will take at least 20min to complete ...
root@ubuntu2204(base) $   conda env list
# conda environments:
#
base                  *  /root/miniconda3
mixmatch                 /root/miniconda3/envs/mixmatch
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   conda env list
# conda environments:
#
base                     /root/miniconda3
mixmatch              *  /root/miniconda3/envs/mixmatch
```

### Level One: Reproduce the Figures

The most straightforward way to reproduce the figures we present in our paper is by running the Jupyter Notebooks again that compile these figures from the Receiver Operating Characteristic (ROC) curve points that we provide within this repository:
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2
root@ubuntu2204(mixmatch) $   ./1_result_figures/generate_all.sh
```

Mind that the names of the figures produced above are not immediately connected to sections in our paper. Please make this mapping yourself.

### Level Two: Train/Evaluate Statistical Classifier and DL Models on Prepared Datasets

Follow the steps below in order to set up the entire environment necessary to train and evaluate our statistical, drift, and shape classifier on the above-listed prepared datasets. Mind that this step comes with heavy hardware expectations (see section above) and will take multiple days per each experiment (`baseline`, `no-cover`, `low-delay`, `high-delay`, `two-to-one`, `live-nym`).
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch
root@ubuntu2204(mixmatch) $   mkdir -p ~/mixmatch/deeplearning/{datasets,delay_matrices}
root@ubuntu2204(mixmatch) $   mkdir -p ~/mixmatch/statistical/datasets
root@ubuntu2204(mixmatch) $   git clone https://github.com/mixnet-correlation/mixmatch_drift_classifier.git ~/mixmatch/deeplearning/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   git clone https://github.com/mixnet-correlation/mixmatch_shape_classifier.git ~/mixmatch/deeplearning/mixmatch_shape_classifier
root@ubuntu2204(mixmatch) $   git clone https://github.com/mixnet-correlation/mixmatch_statistical_classifier.git ~/mixmatch/statistical/mixmatch_statistical_classifier
```

For the deep learning classifiers, we require a set of three delay matrices per each dataset that we want to analyze. Due to their size (the `train_*` come close to 1 GB), we provide them at the following **OSF repository: [osf.io/m9gbz/](https://osf.io/m9gbz/)**. Follow below steps to download all required files into the respective place in our folder structure. **Mind:** The `wget` command within `./2_exps_1_delay-matrices.sh` downloads a zip file of circa **4.8 GB** size on disk.
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2
root@ubuntu2204(mixmatch) $   ./2_exps_1_delay-matrices.sh
root@ubuntu2204(mixmatch) $   tree -a ~/mixmatch/deeplearning/delay_matrices
/root/mixmatch/deeplearning/delay_matrices
├── baseline
│   ├── test_delay_matrix.npz
│   ├── train_delay_matrix.npz
│   └── val_delay_matrix.npz
├── high-delay
│   ├── test_delay_matrix.npz
│   ├── train_delay_matrix.npz
│   └── val_delay_matrix.npz
├── live-nym
│   ├── test_delay_matrix.npz
│   ├── train_delay_matrix.npz
│   └── val_delay_matrix.npz
├── low-delay
│   ├── test_delay_matrix.npz
│   ├── train_delay_matrix.npz
│   └── val_delay_matrix.npz
└── no-cover
    ├── test_delay_matrix.npz
    ├── train_delay_matrix.npz
    └── val_delay_matrix.npz

5 directories, 15 files
```

Now clone any of the above-listed prepared datasets that you'd like to analyze. We show the process in the following for `baseline`, the other datasets proceed accordingly:
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   git clone https://github.com/mixnet-correlation/dataset_baseline_nym-binaries-1.0.2_deeplearning.git ~/mixmatch/deeplearning/datasets/baseline
root@ubuntu2204(mixmatch) $   git clone https://github.com/mixnet-correlation/dataset_baseline_nym-binaries-1.0.2_statistical.git ~/mixmatch/statistical/datasets/baseline
```

**Side note:** Training the DL models drift and shape can be very time-consuming and resource-intense. In case you'd like to use pretrained models to skip over below training steps for the deep learning classifiers and run the evaluation steps directly, please feel free to use the pretrained models provided in subfolder [`./2_pretrained_deeplearning_models`](./2_pretrained_deeplearning_models) in the appropriate places instead. You can do so by moving the pretrained DL model you would like to use for `get_scores.py` to the respective `results` folder and renaming the model's top-level folder to `model.tf`. As an example for the drift model trained on dataset `low-delay`, download the pretrained model folder [`exp05-drift.tf`](./2_pretrained_deeplearning_models/exp05-drift.tf) to an appropriately named subfolder within your `results` folder. Then, rename the downloaded model folder from `exp05-drift.tf` to `model.tf` and run `get_scores.py` as below but with accordingly adjusted `data` and `results` subfolders as arguments.

The following list of commands will take you through one end-to-end analysis cycle of parsing, training, evaluating, and calculating scores for one dataset with our drift classifier, exemplarily for dataset `baseline`. **Please mind that the full process from first to last command takes on the order of days to complete and requires powerful hardware (see [section above](#hardware-considerations)).**
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/deeplearning/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   python parse.py ../datasets/baseline --delaymatpath ../delay_matrices/baseline --experiment 1
... Takes at least 20min to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python train.py
... Takes on the order of many hours complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/latest/ ./results/latest/
... Takes on the order of some hours to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/latest/
... Takes on the order of 1 hour to complete ...
```

When running the deep learning classifiers on multiple datasets, we recommend to name data and results folders within `~/mixmatch/deeplearning/mixmatch_drift_classifier` explicitly after their respective experiment/dataset/purpose.

Once you have completed the step of running `calculate_roc.py` for one of our deep learning classifiers, file `step_results.csv` in the respective results folder contains all obtained data needed to plot the experiment's ROC figure. The ROC data files starting with `subsampled_` that we provide [for reproducing at level 1 above](#level-one-reproduce-the-figures) are filtered versions of this `step_results.csv` file and very specific to our figures. By appropriately filtering `step_results.csv` into ROC data files with a specific number of samples or of a specific packet type, you can arrive at the ROC data files relevant to the figure you are interested in for plotting.

For the special case of the `two-to-one` experiment that is based on the `baseline` dataset, we start from the `baseline`-trained model and instruct the model at inference time to build and analyze the `two-to-one` dataset ad-hoc in the following way:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/deeplearning/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/A_BASELINE_DATA_FOLDER/ ./results/A_BASELINE_RESULTS_FOLDER/ --two2one_case1   # Semi-matched case
... Takes on the order of some hours to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/A_BASELINE_RESULTS_FOLDER/ --two2one
... Takes on the order of 1 hour to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/A_BASELINE_DATA_FOLDER/ ./results/A_BASELINE_RESULTS_FOLDER/ --two2one_case2   # Unmatched case
... Takes on the order of some hours to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/A_BASELINE_RESULTS_FOLDER/ --two2one
... Takes on the order of 1 hour to complete ...
```

For dataset `baseline` and our shape classifier, run:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/deeplearning/mixmatch_shape_classifier
root@ubuntu2204(mixmatch) $   ln -s ~/mixmatch/deeplearning/delay_matrices/baseline/test_delay_matrix.npz ~/mixmatch/deeplearning/datasets/baseline/test_delay_matrix.npz
root@ubuntu2204(mixmatch) $   ln -s ~/mixmatch/deeplearning/delay_matrices/baseline/train_delay_matrix.npz ~/mixmatch/deeplearning/datasets/baseline/train_delay_matrix.npz
root@ubuntu2204(mixmatch) $   ln -s ~/mixmatch/deeplearning/delay_matrices/baseline/val_delay_matrix.npz ~/mixmatch/deeplearning/datasets/baseline/val_delay_matrix.npz
root@ubuntu2204(mixmatch) $   python parse.py ../datasets/baseline --experiment 1
... Takes at least 20min to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python train.py
... Takes on the order of many hours complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/latest/ ./results/latest/
... Takes on the order of some hours to complete ...
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/latest/
... Takes on the order of 1 hour to complete ...
root@ubuntu2204(mixmatch) $   rm ~/mixmatch/deeplearning/datasets/baseline/test_delay_matrix.npz
root@ubuntu2204(mixmatch) $   rm ~/mixmatch/deeplearning/datasets/baseline/train_delay_matrix.npz
root@ubuntu2204(mixmatch) $   rm ~/mixmatch/deeplearning/datasets/baseline/val_delay_matrix.npz
```

Evaluating our statistical classifier on dataset `baseline` requires the following commands:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   mkdir -p ~/mixmatch/statistical/octave-packages
root@ubuntu2204(mixmatch) $   wget 'https://www.nist.gov/document/detwarev2-1-targz' -O ~/mixmatch/statistical/octave-packages/detware.tar.gz
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/statistical/octave-packages
root@ubuntu2204(mixmatch) $   tar xvfz detware.tar.gz && rm -rf detware.tar.gz
root@ubuntu2204(mixmatch) $   chmod 0755 DETware_v2.1 && chown -R root:root DETware_v2.1 && chmod 0440 DETware_v2.1/*
root@ubuntu2204(mixmatch) $   printf "  addpath ('/root/mixmatch/statistical/octave-packages/DETware_v2.1', '-begin');\n" >> ~/.octaverc
root@ubuntu2204(mixmatch) $   mkdir -p ~/mixmatch/statistical/results/{logs,baseline,no-cover,low-delay,high-delay,two-to-one,live-nym}
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/statistical/mixmatch_statistical_classifier
root@ubuntu2204(mixmatch) $   printf "~/mixmatch/statistical/results\n" > ~/mixmatch/statistical/mixmatch_statistical_classifier/MIXCORR_DATA_PATH.txt
root@ubuntu2204(mixmatch) $   printf "~/mixmatch/statistical/datasets\n" > ~/mixmatch/statistical/mixmatch_statistical_classifier/DATABASES_PATH.txt
root@ubuntu2204(mixmatch) $   python real_data_experiment_parser.py
root@ubuntu2204(mixmatch) $   ./transform_flow_pair_lists.tcsh
root@ubuntu2204(mixmatch) $   ./perform_experiment_real_data_alt_delay_characteristic.tcsh
... Script returns immediately, but launches background tasks that take on the order of days to complete ...
root@ubuntu2204(mixmatch) $   octave
octave:1> process_real_data_alt_delay_characteristic_experiment_results("../results", "baseline", 23)
... Takes some time to complete ...
octave:1> exit
```

For the special case of the `two-to-one` experiment, replace the step of running `./perform_experiment_real_data_alt_delay_characteristic.tcsh` above with the following two commands:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/statistical/mixmatch_statistical_classifier
root@ubuntu2204(mixmatch) $   ./perform_experiment_real_data_alt_delay_characteristic_3parties_unmatched_negatives.tcsh
... Script returns immediately, but launches background tasks that take on the order of days to complete ...
root@ubuntu2204(mixmatch) $   ./perform_experiment_real_data_alt_delay_characteristic_3parties_semimatched_negatives.tcsh
... Script returns immediately, but launches background tasks that take on the order of days to complete ...
```


### Level Three: Collect Raw Datasets Yourself

Given the time and size requirements, we offer already prepared datasets ready for analysis by our deep learning and statistical classifiers (see section `Datasets` above). However, our dataset collection orchestrators are available in dedicated repositories and we provide instructions for how to use them below, in case you want to collect datasets yourself. Mind that these orchestrators assume to be run on the public cloud provider Hetzner (they expect Hetzner's CLI tool `hcloud` to be available and authenticated to a Hetzner Cloud account). However, if you aim to run these against a different cloud provider or even locally, the core logic of what needs to happen the individual machines required for data collection is easily extractable from the scripts.
* **Isolated Setup**: [`data-collection_2_isolated-setup`](https://github.com/mixnet-correlation/data-collection_2_isolated-setup)
* **Live-Network Setup**: [`data-collection_3_live-network-setup`](https://github.com/mixnet-correlation/data-collection_3_live-network-setup)

The Jupyter Notebooks in subfolder [`3_process_raw_datasets`](./3_process_raw_datasets) help you transform a raw dataset collected with the above listed orchestrators into a format that is ready to be consumed by our classifiers:
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/3_process_raw_datasets
```

You can find two Notebooks each per dataset that we present in the paper. If you have the ability to connect to the webserver that Jupyter Lab spawns, you can inspect and run the Notebooks via your browser like so:
```bash
root@ubuntu2204(mixmatch) $   jupyter lab --allow-root
```
