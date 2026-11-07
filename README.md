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
| `sdr/fisher.py` | SH fits, tangent derivatives, Fisher metric with direction-dependent variance |
| `tests/test_model.py` | Verification: variance laws, Fisher formula, compression factor, anechoic limit |
| `experiments/` | E0 anechoic check, E1 DRR sweep, E4 mechanism split, figures |
| `theory/theory.tex` | Theory note: model, propositions, pilot results |
| `results/`, `figures/` | Experiment outputs |

## Running

SONICOM files are read from `../perceptual-hrtf-sampling/dependencies/Sonicom_HRTFs`, or from
`SDR_SONICOM_ROOT` if set.

```
pip install -r requirements.txt
python -m pytest tests
python experiments/e1_drr.py 4
python experiments/e1_plots.py
python experiments/e4_mechanism.py
python experiments/e0_anechoic.py
```

Run from the repository root with it on `PYTHONPATH`. E0 also needs the paper-1 cue pipeline in
`../perceptual-hrtf-sampling/design_metric`, or `SDR_PAPER1_PIPELINE`.
