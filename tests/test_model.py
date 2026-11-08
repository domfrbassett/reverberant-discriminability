"""Verification of the cue statistics and the Fisher metric."""
import numpy as np
import pytest

from sdr import cues, fisher, hrtf, room

rng = np.random.default_rng(1)


@pytest.fixture(scope="module")
def head():
    r, f, H, w = hrtf.dtf_spectra(hrtf.available_subjects()[0])
    return r, f, H, w, cues.CueModel(f)


def test_anechoic_limit(head):
    r, f, H, w, cm = head
    c = cm.cues(room.cross_spectra(H, w, np.inf))
    assert np.allclose(c.var[:, -2], cues.SIGMA_ILD ** 2)
    assert np.allclose(c.var[:, -1], cues.SIGMA_ITD ** 2)
    G, Gm, Gv = fisher.fisher_metric(r, c.q, c.var)
    assert np.abs(Gv).max() == 0


def test_converges_to_anechoic(head):
    r, f, H, w, cm = head
    an = room.cross_spectra(H, w, np.inf)
    G0 = fisher.fisher_metric(r, cm.cues(an).q, cm.cues(an).var)[0]
    c = cm.cues(room.cross_spectra(H, w, 80.0), an)
    G = fisher.fisher_metric(r, c.q, c.var)[0]
    assert np.max(np.abs(G - G0) / np.abs(G0).max()) < 1e-3


def test_monaural_compression(head):
    """Band-level slope shrinks by the local DRR factor D_b / (1 + D_b)."""
    r, f, H, w, cm = head
    k = 10 ** (-0 / 10)
    g11, _, _ = room.diffuse_matrix(H, w)
    S = np.abs(H[:, 0]) ** 2 @ cm.G.T
    Sd = g11 @ cm.G.T
    i, j = 300, 301
    eps = 1e-3
    Hp = H.copy()
    Hp[i] = (1 - eps) * H[i] + eps * H[j]
    lev = lambda X, kk: 5 * np.log10((np.abs(X[i, 0]) ** 2 + kk * g11) @ cm.G.T)
    slope_an = lev(Hp, 0) - lev(H, 0)
    slope_rm = lev(Hp, k) - lev(H, k)
    D = S[i] / (k * Sd)
    assert np.allclose(slope_rm, slope_an * D / (1 + D), rtol=2e-2, atol=1e-9)


def test_ild_variance_monte_carlo():
    """Delta-method variance of log(S1/S2) from complex Gaussian samples with coherence g."""
    n, trials = 400, 4000
    for g in (0.99, 0.8, 0.4):
        z = rng.standard_normal((trials, n, 2)) + 1j * rng.standard_normal((trials, n, 2))
        a = z[..., 0]
        b = g * z[..., 0] + np.sqrt(1 - g ** 2) * z[..., 1]
        est = np.log(np.mean(np.abs(a) ** 2, 1) / np.mean(np.abs(b) ** 2, 1))
        assert np.var(est) == pytest.approx(2 * (1 - g ** 2) / n, rel=0.08)


def test_phase_variance_monte_carlo():
    """Variance of the interaural phase estimate is (1 - g^2) / (2 n g^2)."""
    n, trials = 400, 4000
    for g in (0.95, 0.7, 0.4):
        z = rng.standard_normal((trials, n, 2)) + 1j * rng.standard_normal((trials, n, 2))
        a = z[..., 0]
        b = g * z[..., 0] + np.sqrt(1 - g ** 2) * z[..., 1]
        est = np.angle(np.mean(a * np.conj(b), 1))
        assert np.var(est) == pytest.approx((1 - g ** 2) / (2 * n * g ** 2), rel=0.08)


def test_fisher_monte_carlo():
    """Score covariance of Gaussian cues with parameter-dependent mean and variance."""
    theta, h, trials = 0.3, 1e-5, 200000
    mu = lambda t: np.array([np.sin(t), 2 * t])
    var = lambda t: np.array([0.1 + t ** 2, 0.5 * np.exp(t)])
    x = mu(theta) + np.sqrt(var(theta)) * rng.standard_normal((trials, 2))
    ll = lambda t: -0.5 * (((x - mu(t)) ** 2) / var(t) + np.log(var(t))).sum(1)
    score = (ll(theta + h) - ll(theta - h)) / (2 * h)
    dmu = (mu(theta + h) - mu(theta - h)) / (2 * h)
    dlv = (np.log(var(theta + h)) - np.log(var(theta - h))) / (2 * h)
    G = (dmu ** 2 / var(theta)).sum() + 0.5 * (dlv ** 2).sum()
    assert np.var(score) == pytest.approx(G, rel=0.02)


def test_basis_invariance(head):
    r, f, H, w, cm = head
    c = cm.cues(room.cross_spectra(H, w, 0.0), room.cross_spectra(H, w, np.inf))
    G = fisher.fisher_metric(r, c.q, c.var)[0]
    e1, e2 = fisher.azimuth_vector(r), fisher.elevation_vector(r)
    ok = np.abs(r[:, 2]) < 0.95
    tr = fisher.directional_information(G, r, e1) + fisher.directional_information(G, r, e2)
    assert np.allclose(tr[ok], np.trace(G, axis1=1, axis2=2)[ok], rtol=1e-8)


def test_sh_derivative():
    r = rng.standard_normal((900, 3))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    q = (r[:, 0] * r[:, 2])[:, None]
    J = fisher.field_jacobian(r, q, 4)[:, 0, :]
    grad = np.stack([r[:, 2], np.zeros(len(r)), r[:, 0]], 1)
    grad -= (grad * r).sum(1, keepdims=True) * r
    u1, u2 = fisher.tangent_basis(r)
    assert np.allclose(J, np.stack([(grad * u1).sum(1), (grad * u2).sum(1)], 1), atol=1e-6)


def test_coherence_window(head):
    """Band coherence of a direct path plus one reflection follows the Matern-7/2 window."""
    r, f, H, w, cm = head
    from sdr import auditory as aud
    i, j = 300, 500
    out = []
    for lag in (0.2e-3, 2e-3, 20e-3):
        X = H[i] + 0.7 * np.exp(-2j * np.pi * f * lag) * H[j]
        B = cm.band_spectra(np.abs(X[0:1]) ** 2, np.abs(X[1:2]) ** 2, (X[0] * np.conj(X[1]))[None])
        Bi = [a + 0.49 * b for a, b in zip(cm.band_spectra(*[x[i:i + 1] for x in room.direct_spectra(H)]),
                                            cm.band_spectra(*[x[j:j + 1] for x in room.direct_spectra(H)]))]
        out.append((cm.coherence(*B)[0], cm.coherence(*Bi)[0]))
    k = np.argmin(np.abs(cm.fcc - 1000))
    assert out[2][0][k] == pytest.approx(out[2][1][k], abs=0.05)
    assert abs(out[0][0][k] - out[0][1][k]) > abs(out[2][0][k] - out[2][1][k])


def test_images():
    from sdr import ism
    box = ism.Shoebox((8.0, 6.0, 3.0), 0.2, 3)
    src = np.array([3.0, 2.0, 1.2])
    pos, order = box.images(src)
    assert len(pos) == 63 and list(np.bincount(order)) == [1, 6, 18, 38]
    assert np.allclose(pos[0], src)
    first = pos[order == 1]
    walls = [np.array([-3.0, 2.0, 1.2]), np.array([13.0, 2.0, 1.2]), np.array([3.0, -2.0, 1.2]),
             np.array([3.0, 10.0, 1.2]), np.array([3.0, 2.0, -1.2]), np.array([3.0, 2.0, 4.8])]
    for w_ in walls:
        assert np.min(np.linalg.norm(first - w_, axis=1)) < 1e-12


def test_interpolator_exact_on_grid():
    from sdr import ism
    I = ism.HrtfInterpolator(hrtf.available_subjects()[0])
    assert np.max(np.abs(I(I.r[:-1]) - I.H)) < 1e-10 * np.abs(I.H).max()


def test_ism_anechoic_limit(head):
    """With full absorption and no late field the ISM cross-spectra are the direct DTFs."""
    from sdr import ism
    r, f, H, w, cm = head
    I = ism.HrtfInterpolator(hrtf.available_subjects()[0])
    box = ism.Shoebox((8.0, 6.0, 3.0), 1.0 - 1e-12, 2)
    R, _ = ism.binaural_cross_spectra(box, np.array([4.0, 3.0, 1.2]), r[:10], 1.0, I, late=False)
    D = room.direct_spectra(H[:10])
    assert np.allclose(R[0], D[0], rtol=1e-4) and np.allclose(R[2], D[2], rtol=1e-4, atol=1e-9)


def test_subject_203_excluded():
    assert 203 not in hrtf.available_subjects()
    with pytest.raises(ValueError):
        hrtf.sofa_path(203)


def test_arc_derivative_quadratic():
    th = np.radians(np.arange(-90, 91, 5.0))
    y = np.stack([1 + 2 * th - 3 * th ** 2, np.sin(th)], 1)
    d = fisher.arc_derivative(th, y)
    assert np.allclose(d[:, 0], 2 - 6 * th, atol=1e-10)
    assert np.max(np.abs(d[:, 1] - np.cos(th))) < 0.02


def test_brir_spectra_energy_and_window():
    from sdr import brir
    rng = np.random.default_rng(1)
    ir = rng.standard_normal((3, 2, 9000))
    R = brir.cross_spectra(ir)
    m = int(np.ceil(9000 / hrtf.NFFT))
    full = np.abs(np.fft.rfft(ir, n=hrtf.NFFT * m, axis=-1)) ** 2
    assert np.allclose(R[0].sum(1), full[:, 0].sum(1)) and np.allclose(R[1].sum(1), full[:, 1].sum(1))
    w = brir.direct_window(9000, np.array([500, 600, 700]))
    assert w[0, 500] == 1 and w[0, 500 + int(3.6e-3 * brir.FS)] == 0 and np.all((w >= 0) & (w <= 1))


def test_frozen_seat_matches_ism_at_target():
    from sdr import ism, maps
    I = ism.HrtfInterpolator(hrtf.available_subjects()[0])
    box = ism.Shoebox((8.0, 6.0, 3.0), 0.2, 3)
    p = np.array([3.0, 2.5, 1.2])
    u = np.array([[np.cos(0.4), np.sin(0.4), 0.0]])
    R, _ = ism.binaural_cross_spectra(box, p, u, 1.0, I, 5.6)
    seat = maps.from_shoebox(box, p, p + u[0], np.array([1.0, 0.0, 0.0]))
    F = maps.cross_spectra(seat, u, I, 5.6)
    assert all(np.allclose(a, b, rtol=1e-9, atol=1e-12) for a, b in zip(R, F))


def test_pac1_octave_interpolation():
    from sdr import pac1
    f = np.array([31.5, 63.0, 125.0, 177.0, 8000.0, 16000.0])
    e = np.arange(1.0, 9.0)
    g = pac1.octave_to_grid(e, f)
    assert np.allclose(g[[1, 2, 4]], [1, 2, 8]) and g[0] == 1 and g[-1] == 8 and 2 < g[3] < 3


def test_directivity_grid_weights():
    from sdr.directivity import sphere_weights
    T = np.radians(np.arange(0, 181))
    P = np.radians(np.arange(0, 360))
    TT, PP = np.meshgrid(T[1:-1], P)
    u = np.stack([np.cos(TT).ravel(), (np.sin(TT) * np.sin(PP)).ravel(), (np.sin(TT) * np.cos(PP)).ravel()], 1)
    u = np.vstack([[1, 0, 0], u, [-1, 0, 0]])
    w = sphere_weights(u)
    assert np.isclose(w.sum(), 1) and np.isclose(w[u[:, 0] > 0].sum(), 0.5, atol=2e-3)
