"""Ground truth for the geometric invariants: objects whose answer is known in advance.

Closed polygons in R^3 and sampled surfaces, ported from TabularTopology's
``topology.synthetic``. Knot and link values, for the record: a2 (the Casson
invariant) is 1 on the trefoil, -1 on the figure-eight, (p^2-1)(q^2-1)/24 on the
(p, q) torus knot; the Hopf link has linking number -1 as built here; the (2, 2k)
torus link has linking number k in absolute value; the Borromean rings are pairwise
unlinked with Milnor invariant mu(123) = +-1; the Whitehead link has lk = 0 and
mu(1122) = +-1.
"""
import numpy as np


def _t(n):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def circle(n=200, radius=1.0, centre=(0., 0., 0.)):
    t = _t(n)
    return np.stack([radius * np.cos(t), radius * np.sin(t), 0 * t], 1) + np.asarray(centre)


def trefoil(n=600, mirror=False, scale=1.0):
    t = _t(n)
    z = -np.sin(3 * t)
    return scale * np.stack([np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t),
                             -z if mirror else z], 1)


def figure_eight(n=600):
    t = _t(n)
    return np.stack([(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t),
                     np.sin(4 * t)], 1)


def torus_knot(p, q, n=600, R=2.0, r=1.0):
    t = _t(n)
    return np.stack([(R + r * np.cos(p * t)) * np.cos(q * t),
                     (R + r * np.cos(p * t)) * np.sin(q * t), r * np.sin(p * t)], 1)


def hopf_link(n=200):
    t = _t(n)
    return [np.stack([np.cos(t), np.sin(t), 0 * t], 1),
            np.stack([1 + np.cos(t), 0 * t, np.sin(t)], 1)]


def torus_link(k=2, n=400, R=2.0, r=1.0):
    t = _t(n)
    return [np.stack([(R + r * np.cos(k * t + ph)) * np.cos(t),
                      (R + r * np.cos(k * t + ph)) * np.sin(t),
                      r * np.sin(k * t + ph)], 1) for ph in (0.0, np.pi)]


def borromean_rings(n=400, a=1.618, b=1.0):
    t = _t(n)
    z = np.zeros(n)
    return [np.stack([z, b * np.cos(t), a * np.sin(t)], 1),
            np.stack([a * np.sin(t), z, b * np.cos(t)], 1),
            np.stack([b * np.cos(t), a * np.sin(t), z], 1)]


# The Whitehead link of ``knots.constructors.whitehead_link``: a square weaving over and
# under the two lobes of a figure-eight curve (corner polygons, integer coordinates).
WHITEHEAD_CORNERS = (
    [(4, -4, 2), (4, 4, 2), (4, 4, -2), (-4, 4, -2), (-4, 4, 2), (-4, -4, 2), (-4, -4, -2),
     (4, -4, -2)],
    [(-2, 0, 2), (2, 0, 2), (2, 0, 0), (8, 0, 0), (8, 6, 0), (0, 6, 0), (0, 2, 0), (0, 2, -2),
     (0, -2, -2), (0, -2, 0), (0, -6, 0), (-8, -6, 0), (-8, 0, 0), (-2, 0, 0)],
)


def whitehead_link():
    """The Whitehead link: lk = 0, Sato-Levine invariant beta = -mu-bar(1122) = +-1."""
    return [np.array(c, dtype=float) for c in WHITEHEAD_CORNERS]


def unlink(k=2, n=200, sep=3.0):
    return [circle(n, centre=(sep * i, 0.0, 0.0)) for i in range(k)]


def fibonacci_sphere(n, radius=1.0, centre=(0., 0., 0.)):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
                    1) * radius + np.asarray(centre)


def mobius_band_sample(n=1500, width=0.35, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(-width, width, n)
    return np.stack([(1 + v * np.cos(u / 2)) * np.cos(u), (1 + v * np.cos(u / 2)) * np.sin(u),
                     v * np.sin(u / 2)], 1)


def cylinder_sample(n=1500, width=0.35, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(-width, width, n)
    return np.stack([np.cos(u), np.sin(u), v], 1)


def klein_bottle_sample(n=2500, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    return np.stack([(2 + np.cos(v)) * np.cos(u), (2 + np.cos(v)) * np.sin(u),
                     np.sin(v) * np.cos(u / 2), np.sin(v) * np.sin(u / 2)], 1)


def latitude_loop(n=200, colatitude=0.7):
    phi = _t(n)
    s, c = np.sin(colatitude), np.cos(colatitude)
    return (np.stack([s * np.cos(phi), s * np.sin(phi), c * np.ones(n)], 1),
            float(2 * np.pi * (1 - c)))


def sphere_cap(n=3000, colatitude=0.7, pad=0.35, rng=None):
    rng = rng or np.random.default_rng(0)
    ct = rng.uniform(np.cos(min(np.pi, colatitude + pad)), np.cos(max(0.0, colatitude - pad)), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(np.maximum(0.0, 1 - ct ** 2))
    return np.stack([st * np.cos(phi), st * np.sin(phi), ct], 1)
