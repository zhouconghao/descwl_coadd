import sys
import numpy as np

from descwl_shear_sims.sim import make_sim, get_se_dim, get_coadd_center_gs_pos
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.layout import Layout
from descwl_coadd.coadd import make_coadd, get_coadd_psf_at_position
from descwl_coadd.coadd import get_coadd_psf_bbox
import lsst.geom as geom
import galsim
import pytest

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

psf_dim = 51          # coadd PSF box and the crop used in the comparisons
coadd_dim = 101
pixel_scale = 0.2

# Single-epoch PSF stamp and delta rendering stamp: the same odd size, so the
# two truncation footprints coincide, and larger than the coadd PSF box so the
# Lanczos3 warp of the (rotated) 51-px box has full source coverage:
# ceil(51 * sqrt(2)) + 2 * 3 = 79 -> 81.
SE_PSF_DIM = 81


class CustomGalaxyCatalog(object):
    """
    Catalog that uses an explicit list of galsim objects and (u, v) shifts.

    Parameters
    ----------
    gal_list : list[galsim.GSObject]
        The galaxies to place.
    uv_shift : list[tuple[float, float]]
        Per-galaxy (u, v) shifts in arcsec, same length as gal_list.
    layout : Layout | None
        Optional; only used to carry coadd bbox/world origin metadata.
        Positions are taken strictly from `uv_shift`.
    """
    def __init__(self, *, gal_list, uv_shift, coadd_dim, pixel_scale,
                 simple_coadd_bbox, layout=None):
        if gal_list is None or len(gal_list) == 0:
            raise ValueError("gal_list must be a non-empty list of galsim objects")
        if uv_shift is None or len(uv_shift) != len(gal_list):
            raise ValueError("uv_shift must be provided and match len(gal_list)")
        self.gal_type = 'fixed'
        self._gal_list = list(gal_list)
        self._shifts = [galsim.PositionD(u, v) for (u, v) in uv_shift]

        buff = 0  # ignored since positions are explicit

        if isinstance(layout, str):
            self.layout = Layout(layout, coadd_dim, buff, pixel_scale,
                                 simple_coadd_bbox=simple_coadd_bbox)
        else:
            assert isinstance(layout, Layout)
            self.layout = layout

    def __len__(self):
        return len(self._gal_list)

    def get_objlist(self, *, survey):
        """
        Returns a dict with the same structure as other catalogs.
        The provided GSObjects are used verbatim (no flux remapping).
        """
        indexes = list(range(len(self._gal_list)))
        return {
            "objlist": list(self._gal_list),
            "shifts": list(self._shifts),
            "redshifts": None,
            "indexes": indexes,
        }


def _make_sim(
    rng,
    psf_type=None,
    epochs_per_band=3,
    stars=False,
    dither=True,
    rotate=True,
    bands=["i", "z"],
    coadd_dim=101,
    se_dim=None,
    psf_dim=SE_PSF_DIM,
    bad_columns=False,
    psf_variation_factor=None,
    u_shift=0,
    v_shift=0,
    psf=None,
    stamp_size=SE_PSF_DIM,
    im_dtype=np.float64,
):
    buff = 5

    gal = galsim.DeltaFunction(flux=1.0)
    galaxy_catalog = CustomGalaxyCatalog(
        gal_list=[gal],
        uv_shift=[(u_shift, v_shift)],
        coadd_dim=coadd_dim,
        pixel_scale=pixel_scale,
        simple_coadd_bbox=True,
        layout="no_layout",
    )

    if se_dim is None:
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate)

    if psf is None:
        if psf_type == "ps":
            psf = make_ps_psf(rng=rng, dim=se_dim,
                              variation_factor=psf_variation_factor)
        else:
            psf = make_fixed_psf(psf_type=psf_type)

    if stars:
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=100,
        )
    else:
        star_catalog = None

    return make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=star_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=psf_dim,
        epochs_per_band=epochs_per_band,
        g1=0.00,
        g2=0.00,
        bands=bands,
        psf=psf,
        dither=dither,
        rotate=rotate,
        bad_columns=bad_columns,
        draw_noise=False,
        stamp_size=stamp_size,
        im_dtype=im_dtype,
    )


def get_coadd_res(u_shift, v_shift, dither=False, rotate=False):
    rng = np.random.RandomState(2025)
    bands = ['i']

    sim_data = _make_sim(
        rng=rng,
        psf_type='gauss',
        bands=bands,
        coadd_dim=coadd_dim,
        psf_dim=SE_PSF_DIM,
        dither=dither,
        rotate=rotate,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    exps = sim_data['band_data']['i']
    coadd_wcs = sim_data['coadd_wcs']
    coadd_bbox = sim_data['coadd_bbox']
    psf_dims = (psf_dim, psf_dim)   # coadd PSF box; SE stamps are SE_PSF_DIM

    coadd_dict = make_coadd(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=np.random.RandomState(7),
        remove_poisson=False,
        im_dtype=np.float64,
    )

    # center psf
    cen_psf_img = coadd_dict['coadd_psf_exp'].image.array
    cen_psf_img /= cen_psf_img.sum()

    assert cen_psf_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(cen_psf_img).all()
    np.testing.assert_allclose(cen_psf_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    coadd_wcs = sim_data['coadd_wcs']
    coadd_bbox = sim_data['coadd_bbox']

    coadd_bbox_cen_gs_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )

    world_pos = coadd_bbox_cen_gs_skypos.deproject(
        u_shift * galsim.arcsec,
        v_shift * galsim.arcsec,
    )

    dm_world_pos = geom.SpherePoint(world_pos.ra / galsim.degrees,
                                    world_pos.dec / galsim.degrees,
                                    geom.degrees)
    new_image_pos = coadd_wcs.skyToPixel(dm_world_pos)

    dm_image_pos = geom.Point2D(new_image_pos.x, new_image_pos.y)

    psf_off = get_coadd_psf_at_position(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        image_pos=dm_image_pos,
        rng=np.random.RandomState(13),
        remove_poisson=False,
        im_dtype=np.float64,
    )

    off_img = psf_off.computeKernelImage(dm_image_pos).array
    assert off_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(off_img).all()
    np.testing.assert_allclose(off_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    return sim_data, coadd_dict, cen_psf_img, off_img, exps


def get_crop_bbox(u_shift, v_shift, coadd_wcs, coadd_bbox, psf_dim):
    """Crop bbox centered on the same coadd pixel the PSF reconstruction uses.

    ``get_coadd_psf_at_position`` centers the PSF stamp on
    ``round(coadd_wcs.skyToPixel(world_pos))``. Center the delta crop the same
    way so the two stay aligned even when the injection lands on a half-integer
    pixel -- an analytic ``coadd_cen + shift/scale`` center can round to a
    different pixel there (they straddle the ``floor(x + 0.5)`` boundary),
    producing a spurious 1-pixel offset that a naive PSF-vs-delta comparison
    would flag (dramatically so for elliptical PSFs).
    """
    world_pos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    ).deproject(u_shift * galsim.arcsec, v_shift * galsim.arcsec)
    dm_world_pos = geom.SpherePoint(
        world_pos.ra / galsim.degrees,
        world_pos.dec / galsim.degrees,
        geom.degrees,
    )
    pos = coadd_wcs.skyToPixel(dm_world_pos)
    cen_int = geom.Point2I(
        int(np.floor(pos.x + 0.5)),
        int(np.floor(pos.y + 0.5)),
    )

    return get_coadd_psf_bbox(cen_int, psf_dim)


def _report_diff(img, ref, *, u_shift=0.0, v_shift=0.0, dither, rotate,
                 psf_variation_factor=None):
    """Print the max abs difference between two images in a common format
    shared by all tests in this file."""
    max_abs_diff = np.abs(img - ref).max()
    frac_of_peak = max_abs_diff / ref.max()
    # express the fractional difference as a magnitude: dm = 2.5/ln(10) * df/f
    mmag = frac_of_peak * 2.5 / np.log(10) * 1000.0
    print(
        f"shift=({u_shift:+.3f}, {v_shift:+.3f}) "
        f"dither={dither} rotate={rotate} var={psf_variation_factor}: "
        f"max|diff|={max_abs_diff:.3e} "
        f"({frac_of_peak:.2%} of peak, {mmag:.3g} mmag)"
    )
    return max_abs_diff, frac_of_peak


np.random.seed(42)
random_shifts = [
    (np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0))
    for _ in range(3)
]

shift_cases = [
    (0.0, 0.0),
    (10.0 * 0.2, 0.0),
    (10.5 * 0.2, 5.0 * 0.2)
] + random_shifts

dither_rotate_cases = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]

test_cases = [
    (u, v, dither, rotate)
    for (u, v) in shift_cases
    for (dither, rotate) in dither_rotate_cases
]


@pytest.mark.parametrize("dither, rotate", dither_rotate_cases)
def test_center_psf_make_coadd_vs_at_position(dither, rotate):
    rng = np.random.RandomState(2025)

    sim_data = _make_sim(
        rng=rng,
        psf_type='gauss',
        bands=['i'],
        coadd_dim=coadd_dim,
        psf_dim=SE_PSF_DIM,
        dither=dither,
        rotate=rotate,
    )

    exps = sim_data['band_data']['i']
    coadd_wcs = sim_data['coadd_wcs']
    coadd_bbox = sim_data['coadd_bbox']
    psf_dims = (psf_dim, psf_dim)   # coadd PSF box; SE stamps are SE_PSF_DIM

    coadd_dict = make_coadd(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=np.random.RandomState(7),
        remove_poisson=False,
        im_dtype=np.float64,
    )

    coadd_cen = coadd_bbox.getCenter()
    center_pos = geom.Point2D(coadd_cen.x, coadd_cen.y)

    make_coadd_psf_img = (
        coadd_dict['coadd_exp']
        .getPsf()
        .computeKernelImage(center_pos)
        .array
    )

    at_pos_psf = get_coadd_psf_at_position(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        image_pos=center_pos,
        rng=np.random.RandomState(13),
        remove_poisson=False,
        im_dtype=np.float64,
    )
    at_pos_psf_img = at_pos_psf.computeKernelImage(center_pos).array

    _report_diff(
        at_pos_psf_img, make_coadd_psf_img, dither=dither, rotate=rotate,
    )

    np.testing.assert_allclose(at_pos_psf_img, make_coadd_psf_img, atol=1e-15)


@pytest.mark.parametrize("u_shift, v_shift, dither, rotate", test_cases)
def test_coadd_off_cen(u_shift, v_shift, dither, rotate):
    sim_data, coadd_dict, cen_psf_img, off_img, exps = get_coadd_res(
        u_shift, v_shift, dither=dither, rotate=rotate
    )

    crop_box = get_crop_bbox(
        u_shift=u_shift,
        v_shift=v_shift,
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dim=psf_dim,
    )

    crop_image = coadd_dict["coadd_exp"][crop_box].image.array
    crop_image_norm = crop_image / crop_image.sum()

    _report_diff(
        off_img, crop_image_norm, u_shift=u_shift, v_shift=v_shift,
        dither=dither, rotate=rotate,
    )

    if u_shift == 0.0 and v_shift == 0.0:
        # check if the center psf from the old method
        # and new method are the same
        np.testing.assert_allclose(off_img, cen_psf_img, atol=1e-6)

    # check if the off_center psf agree with the input delta object
    np.testing.assert_allclose(off_img, crop_image_norm, atol=1e-6)


# --- varying-PSF delta-function test -----------------------------------------
# With matched SE_PSF_DIM stamps for the delta and the PSF, float64 end to end,
# and the DM PSF sampling the field at the galsim position, the coadded delta
# and the coadd PSF agree to ~1e-11; the remaining floor is the ~1e-10 px
# disagreement between the galsim and DM inverse TAN WCS.
VARYING_PSF_ATOL = 1e-9
CONST_MOFFAT_ATOL = 1e-9

np.random.seed(99)
varying_psf_shifts = [
    (np.random.uniform(-1.8, 1.8), np.random.uniform(-1.8, 1.8))
    for _ in range(5)
]


def _recon_psf_at(sim_data, u_shift, v_shift):
    """Coadd a sim and return the coadd dict plus the normalized coadd-PSF
    image reconstructed at the (u, v) injection position."""
    exps = sim_data["band_data"]["i"]
    coadd_wcs = sim_data["coadd_wcs"]
    coadd_bbox = sim_data["coadd_bbox"]
    psf_dims = (psf_dim, psf_dim)   # coadd PSF box; SE stamps are SE_PSF_DIM

    coadd_dict = make_coadd(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        rng=np.random.RandomState(7),
        remove_poisson=False,
        im_dtype=np.float64,
    )

    # world position of the injected delta from its (u, v) offset, then the
    # corresponding coadd pixel position.
    coadd_cen_skypos = get_coadd_center_gs_pos(
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
    )
    world_pos = coadd_cen_skypos.deproject(
        u_shift * galsim.arcsec,
        v_shift * galsim.arcsec,
    )
    dm_world_pos = geom.SpherePoint(
        world_pos.ra / galsim.degrees,
        world_pos.dec / galsim.degrees,
        geom.degrees,
    )
    image_pos = coadd_wcs.skyToPixel(dm_world_pos)
    dm_image_pos = geom.Point2D(image_pos.x, image_pos.y)

    psf_off = get_coadd_psf_at_position(
        exps=exps,
        coadd_wcs=coadd_wcs,
        coadd_bbox=coadd_bbox,
        psf_dims=psf_dims,
        image_pos=dm_image_pos,
        rng=np.random.RandomState(13),
        remove_poisson=False,
        im_dtype=np.float64,
    )
    off_img = psf_off.computeKernelImage(dm_image_pos).array
    off_img = off_img / off_img.sum()

    return coadd_dict, off_img


def _delta_vs_psf(sim_data, u_shift, v_shift):
    """Return the two quantities we always compare: the normalized coadd delta
    image stamp and the normalized coadd PSF, both at the injection position.

    We compare the coadded delta image against the coadd PSF (never one
    reconstructed PSF against another). PSF-to-PSF comparisons pick up the
    galsim (1-based) vs DM (0-based) 1-pixel convention in where the PSF field
    is sampled during reconstruction; the coadd-delta-vs-coadd-PSF comparison
    is the physically meaningful, robust check.
    """
    coadd_dict, off_img = _recon_psf_at(sim_data, u_shift, v_shift)

    crop_box = get_crop_bbox(
        u_shift=u_shift,
        v_shift=v_shift,
        coadd_wcs=sim_data["coadd_wcs"],
        coadd_bbox=sim_data["coadd_bbox"],
        psf_dim=psf_dim,
    )
    crop_image = coadd_dict["coadd_exp"][crop_box].image.array
    crop_image = crop_image / crop_image.sum()

    return off_img, crop_image


@pytest.mark.parametrize("psf_variation_factor", [1, 3])
@pytest.mark.parametrize("dither, rotate", [(False, False), (True, True)])
@pytest.mark.parametrize("u_shift, v_shift", varying_psf_shifts)
def test_delta_matches_varying_psf(
    u_shift, v_shift, dither, rotate, psf_variation_factor
):
    """The coadded delta image equals the coadd PSF at its location, for a
    spatially varying (ps) PSF."""
    sim_data = _make_sim(
        rng=np.random.RandomState(2025),
        psf_type="ps",
        psf_variation_factor=psf_variation_factor,
        bands=["i"],
        coadd_dim=coadd_dim,
        psf_dim=SE_PSF_DIM,
        dither=dither,
        rotate=rotate,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    off_img, crop_image = _delta_vs_psf(sim_data, u_shift, v_shift)

    # the reconstructed PSF is a valid, normalized stamp
    assert off_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(off_img).all()
    np.testing.assert_allclose(off_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    _report_diff(
        off_img, crop_image, u_shift=u_shift, v_shift=v_shift,
        dither=dither, rotate=rotate,
        psf_variation_factor=psf_variation_factor,
    )

    # delta * PSF == PSF: the imaged point source matches the coadd PSF at
    # that position, even though the PSF varies across the field.
    np.testing.assert_allclose(off_img, crop_image, atol=VARYING_PSF_ATOL)


@pytest.mark.parametrize("psf_variation_factor", [1, 3])
@pytest.mark.parametrize("dither, rotate", [(False, False), (True, True)])
@pytest.mark.parametrize("u_shift, v_shift", varying_psf_shifts[:3])
def test_delta_vs_psf_varying_and_constant(
    u_shift, v_shift, dither, rotate, psf_variation_factor
):
    """Side by side: a varying (ps) PSF and a constant PSF frozen from the
    varying PSF's value at the injection. For each, the coadded delta image must
    match the coadd PSF at the injection.

    We deliberately compare the coadd delta image to the coadd PSF for each PSF
    rather than comparing the two reconstructed PSFs to each other: a
    PSF-to-PSF comparison picks up the galsim (1-based) vs DM (0-based) 1-pixel
    convention in where the varying PSF field is sampled during reconstruction.
    """
    seed = 2025
    se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate)

    # build the varying psf with its own rng so the sim rng can be reused
    ps_psf = make_ps_psf(
        rng=np.random.RandomState(seed + 1),
        dim=se_dim,
        variation_factor=psf_variation_factor,
    )

    sim_var = _make_sim(
        rng=np.random.RandomState(seed),
        psf=ps_psf,
        bands=["i"],
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=SE_PSF_DIM,
        dither=dither,
        rotate=rotate,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    # freeze a constant PSF = the varying field at the injection's SE position
    coadd_cen_skypos = get_coadd_center_gs_pos(
        coadd_wcs=sim_var["coadd_wcs"],
        coadd_bbox=sim_var["coadd_bbox"],
    )
    world_pos = coadd_cen_skypos.deproject(
        u_shift * galsim.arcsec,
        v_shift * galsim.arcsec,
    )
    const_psf = ps_psf.getPSF(sim_var["se_wcs"]["i"][0].toImage(world_pos))

    # identical sim except the PSF is fixed to const_psf (same sim rng)
    sim_const = _make_sim(
        rng=np.random.RandomState(seed),
        psf=const_psf,
        bands=["i"],
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        psf_dim=SE_PSF_DIM,
        dither=dither,
        rotate=rotate,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    # each PSF: the coadded delta image must match the coadd PSF
    for sim_data in (sim_var, sim_const):
        off_img, crop_image = _delta_vs_psf(sim_data, u_shift, v_shift)
        assert np.isfinite(off_img).all()
        np.testing.assert_allclose(off_img.sum(), 1.0, rtol=1e-6, atol=1e-8)
        _report_diff(
            off_img, crop_image, u_shift=u_shift, v_shift=v_shift,
            dither=dither, rotate=rotate,
            psf_variation_factor=psf_variation_factor,
        )
        np.testing.assert_allclose(off_img, crop_image, atol=VARYING_PSF_ATOL)


@pytest.mark.parametrize("epochs_per_band", [1, 3])
@pytest.mark.parametrize("dither, rotate", [(False, False), (True, True)])
@pytest.mark.parametrize("u_shift, v_shift", varying_psf_shifts[:3])
def test_delta_matches_constant_sheared_moffat(
    u_shift, v_shift, dither, rotate, epochs_per_band
):
    """Delta-function injection test with a constant *sheared Moffat* PSF.

    The coadded delta image must match the coadd PSF at the injection. Being a
    constant PSF there is no spatial field to sample, so the galsim/DM 1-pixel
    convention is irrelevant and this holds even with dither/rotate -- the only
    floor is the Moffat-wing truncation at the psf_dim stamp. Checked for both a
    single epoch and a 3-epoch stack.
    """
    psf = galsim.Moffat(beta=2.5, fwhm=0.8).shear(g1=0.05, g2=-0.03)

    sim_data = _make_sim(
        rng=np.random.RandomState(2025),
        psf=psf,
        bands=["i"],
        coadd_dim=coadd_dim,
        psf_dim=SE_PSF_DIM,
        epochs_per_band=epochs_per_band,
        dither=dither,
        rotate=rotate,
        u_shift=u_shift,
        v_shift=v_shift,
    )

    off_img, crop_image = _delta_vs_psf(sim_data, u_shift, v_shift)

    assert off_img.shape == (psf_dim, psf_dim)
    assert np.isfinite(off_img).all()
    np.testing.assert_allclose(off_img.sum(), 1.0, rtol=1e-6, atol=1e-8)

    _report_diff(
        off_img, crop_image, u_shift=u_shift, v_shift=v_shift,
        dither=dither, rotate=rotate,
    )

    np.testing.assert_allclose(off_img, crop_image, atol=CONST_MOFFAT_ATOL)
