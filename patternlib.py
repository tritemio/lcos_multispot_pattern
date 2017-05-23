"""
This module contains functions to compute the multispot LCOS pattern.

The main top-level function to call to obtain a complete pattern is
:func:`compute_pattern`.

Unless specified otherwise, coordinates are in LCOS pixel units.

"""


import numpy as np


# LCOS constants
LCOS_X_SIZE, LCOS_Y_SIZE, LCOS_PIX_SIZE = 800, 600, 20e-6

# LCOS mesh-grid of pixels. Allocated once when the module is imported
YL, XL = np.mgrid[:LCOS_Y_SIZE, :LCOS_X_SIZE]


def black_pattern(dtype=np.uint8):
    return np.zeros((LCOS_Y_SIZE, LCOS_X_SIZE), dtype=dtype)


def phase_spherical(r, f, wavelen=532e-9):
    """Phase which converts a plane wave to a converging spherical wave.

    This returns the phase transformation (in units of pi) applied by an ideal
    lens of focal length `f` as a function of the distance `r` from the
    optical axis. The exact phase is computed (no paraxial approximation).

    Arguments:
        r (float or array): distance from the optical axis
        f (float): the focal length of the ideal lens (i.e. radius of the
            spherical wave).

    Returns:
        Phase in pi units for each input distance `r`.
    """
    return -(2 / wavelen) * (np.sqrt(r**2 + f**2) - f)


def get_steer_pattern(lw, vmax=255, horizontal=True):
    """Horizontal or vertical pattern for beam steering.

    Arguments:
        lw (uint): line-width in LCOS pixels for the steering pattern
        vmax (int, 0..255): max value for the steering pattern
        horizontal (bool): if True draw horizontal lines, else vertical.

    Returns:
        2D array (uint8) same size as the LCOS containing the steering pattern.
    """
    a = black_pattern()
    if vmax > 0:
        assert lw > 0
        assert vmax <= 255
        row_wise_a = a
        if not horizontal:
            row_wise_a = a.T
        p = np.zeros(row_wise_a.shape[0])
        for i in range(lw):
            p[i::2*lw] = vmax
        row_wise_a[:] = p[:, np.newaxis]
    return a


def spotmap_func(x, y, pitch_xy, nspots_xy):
    """Return the spot number containing the input point(s).

    Arguments:
        x, y (arrays of floats): coordinates of the input points
            to be labeled with a spot number. The coordinates needs to be
            in the LCOS frame of reference with origin in the LCOS center.
        pitch_xy (tuple of 2 floats): X and Y pattern pitch in LCOS pixels.
        nspots_xy (tuple of 2 ints): number of spots in X and Y direction.

    Returns:
        Return an array with same shape as `x` (or `y`) containing an integer
        label (the spot number) for each input point. If no spot contains
        a given point the value will be NaN.
    """
    assert x.shape == y.shape
    spotnum = [0, 0]
    for i, (v, pitch, nspots) in enumerate(zip((x, y), pitch_xy, nspots_xy)):
        offset = 0 if (nspots % 2) == 0 else 0.5
        spotnum[i] = np.floor(v / pitch + offset)
        smin, smax = -(nspots // 2), (nspots // 2) + nspots % 2 - 1
        spotnum[i][(spotnum[i] < smin) + (spotnum[i] > smax)] = np.nan
    Z = spotnum[0] + spotnum[1] * nspots_xy[0]
    Z -= np.nanmin(Z)
    return Z


def pitch_from_centers(X, Y):
    """Spot pitch in X and Y direction estimated from spot centers (X, Y).
    """
    assert X.shape == Y.shape
    assert X.size > 1
    nspots_y, nspots_x = X.shape
    if nspots_x > 1 and nspots_y == 1:
        pitch_x = pitch_y = np.mean(np.diff(X, axis=1))
    elif nspots_y > 1 and nspots_x == 1:
        pitch_x = pitch_y = np.mean(np.diff(Y, axis=0))
    else:
        # both nspots_x and nspots_y are > 1
        pitch_x = np.mean(np.diff(X, axis=1))
        pitch_y = np.mean(np.diff(Y, axis=0))
    return pitch_x, pitch_y


def get_spot_regions(nspots_x, nspots_y, pitch_x, pitch_y,
                     center_x=0, center_y=0, rotation=0):
    """Compute the rectangular region of the LCOS image containing each spot.

    For details of the algorithm see notebook:

    - Pattern Roto-Translation Documentation.ipynb

    Arguments:
        nspots_x, nspots_y (ints): number of spots in the X and Y direction.
        pitch_x, pitch_y (floats): spot pitch in X and Y direction.
        center_x, center_y (floats): coordinate of the pattern center.
        rotation (float): pattern rotation angle in degree.

    Returns:
        2D array of labels (ints) on the LCOS frame of reference. Spots are
        numbered starting from 0 and going through columns first.
    """
    # NOTE: XL, YL are coordinates of LCOS pixels with origin in the corner.
    #       center_x, center_y are the coordinates of the pattern center with
    #       respect to the LCOS center.
    XLtr, YLtr = rotate((XL - LCOS_X_SIZE // 2 - center_x),
                        (YL - LCOS_Y_SIZE // 2 - center_y), angle=-rotation)
    spot_regions = spotmap_func(XLtr, YLtr, (pitch_x, pitch_y),
                                (nspots_x, nspots_y))
    return spot_regions


def single_spot_pattern(xm, ym, mask=None, a=None, f=30e-3, wavelen=532e-9,
                        phase_max=0, stretchx=1.):
    """Single spot lens and linear steering phase pattern (1 = pi).

    Arguments:
        xm, ym (floats): coordinates of the spot center in LCOS pixel units.
            Origin of coordinates is on the top left LCOS corner.
        mask (2D bool array): mask selecting the extension of the spot. The
            mask is usually True on a rectangular region around the spot center.
        a (2D array): an optional 2D array in which the pattern is written.
            Only the region defined by `mask` will be modified.
        f (float): focal length of the lens created on the phase pattern
            and used to focus a plane wave into a spot.
        wavelen (float): wavelength of the input laser.
        phase_max (float): a constant phase to add to the spherical phase
            of the spot. Since the spherical phase is 0 at its maximum
            (i.e. at the center xm, ym), the value `phase_max` is the max
        stretchx (float): scaling factor for the X coordinates of each single
            spot pattern.

    Returns:
        2D array containing a single spot. If `a` is passed it is modified
        in-place adding a single-spot pattern in the region defined by mask.
    """
    if a is None:
        a = black_pattern(float)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    radius = lambda x, y: np.sqrt((x * stretchx)**2 + y**2)
    R = radius((XL[mask] - xm) * LCOS_PIX_SIZE,
               (YL[mask] - ym) * LCOS_PIX_SIZE)
    a[mask] = phase_max + phase_spherical(R, f=f, wavelen=wavelen)
    return a


def multispot_pattern(Xm, Ym, labels, phase_max, f=30e-3, wavelen=532e-9,
            phase_factor=1, phase_wrap_pos=False, phase_wrap_neg=True,
            stretchx=1, dtype=np.uint8):
    """Pattern for spots centered in X,Y and rectangular limits defined in C.

    Arguments:
        Xm, Ym (2D arrays): coordinates of spot centers with respect to the
            LCOS center. The Xm and Ym shape (array's rows x cols) should
            be the same as the spot grid shape (pattern's rows x cols).
        labels (2D array): array of spot labels (ints) starting from 0.
            Defines rectangular regions for each spot on the LCOS image.
        f (float): focal length of the lens created on the phase pattern
            and used to focus a plane wave into a spot.
        wavelen (float): wavelength of the input laser.
        phase_max (float): constant phase added to the pattern (in pi units).
            See :func:`single_spot_pattern` for details.
        phase_factor (uint8): the 8-bit value [0..255] corresponding to pi
        phase_wrap_neg (bool): if True wraps all the negative-phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.
            When False, the negative phase values are set to 0.
        phase_wrap_pos (bool): if True, wrap the positive phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.
        stretchx (float): scaling factor for the X coordinates of each single
            spot pattern.
        dtype (numpy.dtype): data type to use in the returned array.
            Default uint8.

    Returns:
        A 2D array containing phase pattern image for the defined spots.
    """
    X = Xm + LCOS_X_SIZE // 2
    Y = Ym + LCOS_Y_SIZE // 2

    a = black_pattern(float)
    for ispot, (xm, ym) in enumerate(zip(X.ravel(), Y.ravel())):
        mask = labels == ispot
        single_spot_pattern(xm, ym, mask=mask, a=a, phase_max=phase_max,
                            f=f, wavelen=wavelen, stretchx=stretchx)

    a = phase_wrapping(a, phase_max=phase_max, phase_factor=phase_factor,
                       phase_wrap_pos=phase_wrap_pos,
                       phase_wrap_neg=phase_wrap_neg)
    return a.round().astype(dtype)


def phase_wrapping(a, phase_max, phase_factor, phase_wrap_pos, phase_wrap_neg):
    """Wrap and scale phase according to input parameters.

    Arguments:
        phase_max (float): peak phase value reached in the pattern.
        phase_factor (uint8): the 8-bit value [0..255] corresponding to pi
        phase_wrap_neg (bool): if True wraps all the negative-phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.
            When False, the negative phase values are set to 0.
        phase_wrap_pos (bool): if True, wrap the positive phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.

    Returns:
        A copy of input "phase" array `a` with applied wrapping and scaling.
    """
    if phase_wrap_neg or phase_wrap_pos:
        # smallest multiple of 2 contained in phase_max
        phase_wrap_max = 2 if phase_max <= 2 else (phase_max // 2) * 2

    if phase_wrap_pos:
        pos_phase = a > 0
        # wrap phase between 0 and phase_wrap_max (in pi units)
        a[pos_phase] = a[pos_phase] % phase_wrap_max

    neg_phase = a < 0
    if phase_wrap_neg:
        # wrap phase between 0 and phase_wrap_max (in pi units)
        a[neg_phase] = a[neg_phase] % phase_wrap_max
    else:
        a[neg_phase] = 0

    a *= phase_factor
    return a


def compute_mspotpattern(Xm, Ym, lens_params, steer_params, sparams=None, pad=2,
                  ref_spot=4, stretch=False,
                  ref_spot_dark=False, dark_all=False, nospot=False):
    """Return the pattern with the multi-spot lenses and the beam steering.

    Arguments:
        Xm, Ym (2D arrays): coordinates of spot centers with respect to the
            LCOS center. The Xm and Ym shape (array's rows x cols) should
            be the same as the spot spot grid shape (pattern's rows x cols).
        lens_params (dict): parameters for the multispot pattern.
            See :func:`multispot_pattern`.
        steer_params (dict): parameters for the beam steering pattern.
            See :func:`get_steer_pattern`.
        pad (uint): # pixels of zero-padding around the lens pattern before
            the steering pattern starts.
        ref_spot (int): index of the spot considered as reference (e.g. center).
        ref_spot_dark (bool): if True darken the reference spot.
        stretch (bool): if True and `pitch_x != pitch_y` stretch the single
            spot pattern to match the pitch_x/y ratio. The stretching is
            simply a linear rescaling of the X coordinates, the Y coordinates
            are left unchanged. If False the single spot pattern is circular.

    Returns:
        A 2D array containing the complete phase pattern image with both spots
        and beam steering pattern.
    """
    XM, YM = np.atleast_2d(Xm), np.atleast_2d(Ym)
    assert len(XM.shape) == len(YM.shape) == 2

    if sparams is None:
        nspots_x, nspots_y = XM.shape
        pitch_x, pitch_y = pitch_from_centers(XM, YM)
        sparams = dict(nspots_x=nspots_x, nspots_y=nspots_y,
                       pitch_x=pitch_x, pitch_y=pitch_y,
                       center_x=XM.ravel().mean(), center_y=YM.ravel().mean())
    spot_regions = get_spot_regions(**sparams)
    if stretch:
        stretchx = sparams['pitch_y'] / sparams['pitch_x']
    else:
        stretchx = 1

    if ref_spot_dark:
        if ref_spot >= 0 and ref_spot < XM.size:
            spot_regions[spot_regions == ref_spot] = np.nan
        else:
            print('WARNING: ref_spot out of range: %d' % ref_spot)
    a = multispot_pattern(XM, YM, spot_regions, dtype=np.uint8,
                          stretchx=stretchx, **lens_params)

    if steer_params['vmax'] > 0:
        # NOTE: pad is ignored here
        steer_img = get_steer_pattern(**steer_params)
        mask = np.isnan(spot_regions)
        a[mask] = steer_img[mask]
    return a


def compute_linepattern(center, width, horiz, f, phase_max, wavelen,
                        phase_factor, phase_wrap_neg, phase_wrap_pos,
                        steer_params, pad=0):
    """Compute a line-pattern with an optional steer pattern.

    Arguments:
        f (float): focal length of the cylindrical lens
            used to focus a plane wave into a line.
        wavelen (float): wavelength of the input laser.
        phase_max (float): peak phase reached in the middle of the lens pattern
            (in pi units).
        phase_factor (uint8): the 8-bit value [0..255] corresponding to pi
        phase_wrap_neg (bool): if True wraps all the negative-phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.
            When False, the negative phase values are set to 0.
        phase_wrap_pos (bool): if True, wrap the positive phase values into
            [0..phase_wrap_max]. phase_wrap_max is 2 when `phase_max` <= 2,
            otherwise is the smallest multiple of 2 contained in `phase_max`.
        steer_params (uint): dict of beam-steering parameters passed to
            :func:`get_steer_pattern`.
        pad (int): padding in pixels between the line pattern and the steering
            pattern. Default 0.

    Returns:
        A 2D array containing the complete phase pattern image with both
        the cylindrical lens (line pattern) and beam steering pattern.
    """
    a = black_pattern(float)
    if horiz:
        size_cross = LCOS_Y_SIZE
        a_rot = a
    else:
        size_cross = LCOS_X_SIZE
        a_rot = a.T

    delta = width / 2
    xy = np.arange(size_cross) - size_cross // 2
    mask = np.abs(xy - center) < delta
    r = (xy[mask] - center) * LCOS_PIX_SIZE
    phase = phase_max + phase_spherical(r, f=f, wavelen=wavelen)
    a_rot[mask] = phase[:, np.newaxis]
    a = phase_wrapping(a, phase_max=phase_max, phase_factor=phase_factor,
                       phase_wrap_pos=phase_wrap_pos,
                       phase_wrap_neg=phase_wrap_neg)

    if steer_params['vmax'] > 0:
        spattern = get_steer_pattern(**steer_params)
        if not horiz:
            spattern = spattern.T
        smask = np.abs(xy - center) >= delta + pad
        a_rot[smask] = spattern[smask]

    return a.round().astype('uint8')


def spot_coord_grid(nspots_x, nspots_y, pitch_x=25, pitch_y=25,
                    center_x=0, center_y=0, rotation=0):
    """Returns the coordinates of spots arranged on a rectangular grid.

    Arguments:
        nspots_x, nspots_y (ints): number of spots in the X and Y direction.
        pitch_x, pitch_y (floats): spot pitch in X and Y direction.
        center_x, center_y (floats): coordinate of the pattern center.
        rotation (float): pattern rotation angle in degree.

    Returns:
        A tuple (X, Y) of two 2D arrays containing the grid of spot centers
        coordinates with respect to the LCOS center and in LCOS pixel units.
        These arrays can be directly passed to :func:`phase_pattern` to
        generate a pattern of spots.
    """
    xp = (np.arange(0, nspots_x, dtype=float) - (nspots_x-1)/2) * pitch_x
    yp = (np.arange(0, nspots_y, dtype=float) - (nspots_y-1)/2) * pitch_y
    Xp, Yp = np.meshgrid(xp, yp)  # spot centers in pattern space

    # Roto-translation to go to LCOS space
    Xm, Ym = rotate(Xp, Yp, rotation)
    Xm += center_x
    Ym += center_y
    return Xm, Ym


def rotate(x, y, angle):
    """Rotate the point (x, y) (scalars or arrays) with respect to the origin.

    Arguments:
        x, y (floats or arrays): input coordinates to be transformed.
        angle (float): rotation angle in degrees. When the Y axis points
            up and the X axis points right, a positive angle result in
            a counter-clock-wise rotation.

    Returns:
        New coordinates or the rotated point.
    """
    if angle == 0:
        return x, y
    shape = x.shape
    assert shape == y.shape
    x_ = x.ravel()
    y_ = y.ravel()
    theta = angle * np.pi / 180
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    v = np.vstack([x_, y_])
    xr, yr = rot_matrix @ v
    return xr.reshape(shape), yr.reshape(shape)


def get_test_pattern():
    """Return a test pattern.
    """
    a = np.arange(LCOS_Y_SIZE * LCOS_X_SIZE, dtype=float)
    a *= (255 / a.max())
    a[:256] = np.arange(256)
    a[LCOS_X_SIZE:LCOS_X_SIZE + 256] = np.arange(256)[::-1]
    return a.astype('uint8').reshape(LCOS_Y_SIZE, LCOS_X_SIZE)


def spot_coord_test():
    """Return a set of spot coordinates to used for testing.
    """
    step = 50
    Xm = np.arange(-200, 200, step, dtype=float)
    Ym = 10 * np.cos(Xm * 2*np.pi / (4*step))
    return Xm, Ym


def sanitize_spot_coord(Xm, Ym):
    """Make sure the spot coordinates arrays are consistent.
    """
    if len(Xm) == 0 or len(Ym) == 0:
        print('WARNING: At lest one spot coordinate is empy.')
        return spot_coord_test()
    elif len(Xm) != len(Ym):
        print('WARNING: X and Y spot coordinates have different sizes.')
        return spot_coord_test()

    Xm = np.array(Xm)
    Ym = np.array(Ym)
    return Xm, Ym


def compute_pattern(ncols, nrows, rotation, spotsize, pitch_x, pitch_y,
                    center_x, center_y, wavelen, steer_lw, steer_vmax,
                    ref_spot, focal, phase_max, phase_factor, steer_pad,
                    Xm, Ym, phase_wrap_pos, phase_wrap_neg, steer_horiz,
                    test_pattern, grid, steer_only, dark_all, ref_spot_dark,
                    stretch):
    """Return a complete LCOS pattern computed from the input parameters.

    This is a small wrapper around the lower-level functions:

    - :func:`compute_linepattern`
    - :func:`compute_mspotpattern`
    - :func:`spot_coord_grid`
    - :func:`multispot_pattern`
    - :func:`get_steer_pattern`

    See these functions for a description of the arguments not documented below.

    Arguments:
        Xm, Ym (1D or 2D arrays): coordinates of spot centers with respect to
            the LCOS center. The Xm and Ym shape (array's rows x cols) should
            be the same as the spot spot grid shape (pattern's rows x cols).
            If the spot pattern has only one row of spots the arrays can be 1D.
            These arguments are only used when `grid = False` and are
            ignored otherwise.
        grid (bool): If False, the spot coordinates are taken from
            from the `Xm` and `Ym` input arguments and the spot grid
            parameters are ignored (`nspots_*`, `pitch_*`, `center_*`,
            `rotation`). If True, compute the spot coordinates from
            spot grid parameters (and `Xm` and `Ym` are ignored).
        center_x, center_y (floats): coordinates of the multispot pattern.
            Used also for line pattern, see explaination below.
        pitch_x, pitch_y (floats): pitch of the mutispot pattern in X or Y
            direction. If one of the pitch is 0 for one direction, the pitch
            and center for the other direction are used as width and center
            of a line pattern (see :func:`compute_linepattern`).
        steer_only (bool): if True return only the steering pattern, ignoring
            all the other arguments.
        dark_all (bool): if True return an array of zeros (constant phase).
        test_pattern (bool): return a test pattern (discarding all the other
            arguments).

    Returns:
        2D array (uint8) containing the complete LCOS pattern.
    """
    steer_params = dict(vmax=steer_vmax, lw=steer_lw,
                        horizontal=steer_horiz)
    if test_pattern:
        return get_test_pattern()
    if dark_all:
        return black_pattern()
    if steer_only:
        return get_steer_pattern(**steer_params)

    sparams = None
    if grid:
        sparams = dict(nspots_x=ncols, nspots_y=nrows,
                       rotation=rotation, pitch_x=pitch_x, pitch_y=pitch_y,
                       center_x=center_x, center_y=center_y)
        Xm, Ym = spot_coord_grid(**sparams)
    else:
        Xm, Ym = sanitize_spot_coord(Xm, Ym)

    lens_params = dict(wavelen=wavelen, f=focal, phase_max=phase_max,
                       phase_factor=phase_factor,
                       phase_wrap_pos=phase_wrap_pos,
                       phase_wrap_neg=phase_wrap_neg)

    line = (pitch_x == 0) or (pitch_y == 0)
    if line:
        if pitch_x == 0:
            center, width, horiz = center_y, pitch_y, True
        elif pitch_y == 0:
            center, width, horiz = center_x, pitch_x, False
        a = compute_linepattern(center=center, width=width, horiz=horiz,
                                steer_params=steer_params, pad=steer_pad,
                                **lens_params)
    else:
        a = compute_mspotpattern(Xm, Ym, lens_params, steer_params,
                        sparams=sparams, pad=steer_pad, stretch=stretch,
                        ref_spot_dark=ref_spot_dark, ref_spot=ref_spot)
    return a
