# Spatial Discriminability in Reverberant Rooms

A Fisher-information theory of how a room limits a listener's ability to tell source
directions apart. The room enters through the expected binaural cross-spectral matrix in each
auditory band; cue means, cue variances and the Fisher metric over source direction follow
from it. The anechoic metric of the HRTF upsampling paper is the limit of no reverberant energy.

## Layout

| Path | Contents |
| --- | --- |
| `sdr/hrtf.py` | SONICOM loading, directional transfer functions, Voronoi quadrature |
| `sdr/auditory.py` | Gammatone power responses, ITD low-pass, Reijniers ITD transform |
| `sdr/cues.py` | Expected cues and interaural variance from cross-spectra |
| `sdr/room.py` | Direct path plus diffuse late field |
| `sdr/fisher.py` | SH fits, tangent derivatives, Fisher metric with direction-dependent variance; 1-D arc information |
| `sdr/ism.py` | Shoebox image sources, HRTF interpolation, precedence weighting |
| `sdr/brir.py` | Surrey measured BRIRs: loading, direct window, cross-spectra on the HRTF grid |
| `sdr/maps.py` | Acuity at a seat from fixed reflections (shoebox or path export) for room maps |
| `sdr/pac1.py` | Seats from converted Pachyderm results |
| `sdr/directivity.py` | Loudspeaker directivity from BRAS magnitude files |
| `tests/test_model.py` | Verification: variance laws, Fisher formula, compression factor, anechoic limit |
| `experiments/` | E0 anechoic check, C1 calibration, E1 DRR sweep, E1b shoebox, E1c ITD estimator, E1d lower cap, E2 precedence, E3 measured rooms, E4 mechanism split, E3b BRAS CR4, E5 redundancy, E6 room maps, figures |
| `data/rakerd_hartmann_2010.csv` | Psychometric functions read from the vector figures of Rakerd & Hartmann (2010) |
| `data/litovsky_shinn_2001_fig3.csv` | Lead, lag and single-burst ITD jnds digitised from Litovsky & Shinn-Cunningham (2001) |
| `data/surrey_brirs/` | Surrey real-room BRIRs (download from github.com/IoSR-Surrey/RealRoomBRIRs; not tracked) |
| `paper/paper.tex` | Paper draft; unfinished sections in red |
| `theory/theory.tex` | Theory note: model, propositions, results |
| `results/`, `figures/` | Experiment outputs |

## Data

Third-party data are not in the repository. Obtain them from their sources and place them as below.

| Data | Source | Location |
| --- | --- | --- |
| SONICOM HRTFs (subject 203 excluded) | SONICOM HRTF dataset | `../perceptual-hrtf-sampling/dependencies/Sonicom_HRTFs`, or set `SDR_SONICOM_ROOT` |
| Surrey real-room BRIRs | github.com/IoSR-Surrey/RealRoomBRIRs | `data/surrey_brirs/RealRoomBRIRs-master/` |
| BRAS CR4 BRIRs and RIRs | BRAS, doi:10.14279/depositonce-6726.3 (`1_scene_descriptions-CR4.zip`) | split with `data/split_bras_cr4.py` into `data/bras_cr4/` |
| QSC K8 directivity | BRAS (`2_source_and_receiver_descriptions-QSC_K8.zip`, file `..._MPS_front_pole.csv`) | `data/bras_cr4/QSC_K8/` |
| Pachyderm results | your own runs, converted with `data/convert_pac1.py` | `data/*.npz` |

The digitised psychometric data of Rakerd & Hartmann (2010) and Litovsky & Shinn-Cunningham (2001)
are included as CSV; the articles are not.

`data/CR4_pachyderm_setup.py` (Rhino) places and switches the Pachyderm sources and receivers of the
BRAS CR4 runs; `data/CR4_place_pachyderm_points.py` places them once.

## Running

SONICOM files are read from `../perceptual-hrtf-sampling/dependencies/Sonicom_HRTFs`, or from
`SDR_SONICOM_ROOT` if set. Subject 203 is corrupted and excluded in `sdr/hrtf.py`.

```
pip install -r requirements.txt
python -m pytest tests
python experiments/c1_calibrate.py
python experiments/e2_digitise.py
python experiments/e2_precedence.py
python experiments/e1_drr.py 4
python experiments/e1_plots.py
python experiments/e4_mechanism.py
python experiments/e0_anechoic.py
python -m experiments.e1b_shoebox 2
python -m experiments.e1b_summary
python -m experiments.e3_measured
python -m experiments.e3_summary
python -m experiments.e3b_cr4
python -m experiments.e3b_summary
python -m experiments.e5_redundancy 2
python -m experiments.e5_analysis
python -m experiments.e1c_itd_mode
python -m experiments.e1d_lower_cap
python -m experiments.e6_frozen_check
python -m experiments.e6_maps 2
python -m experiments.e6_summary
```

Run from the repository root with it on `PYTHONPATH`. E0 also needs the paper-1 cue pipeline in
`../perceptual-hrtf-sampling/design_metric`, or `SDR_PAPER1_PIPELINE`.
