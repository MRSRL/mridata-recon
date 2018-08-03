"""General recon functions for FSE datasets."""
import numpy as np
from mrirecon import fftc

def _compute_coefficients_ahncho(kscalib):
    """Compute correction coefficients for FSE using Ahn-Cho method.

    kscalib:  [..., channels, segments, kx]
    """
    # offset start and end to avoid effects from fft wrap
    istart = 5
    iend = -5

    kscalib_shape = kscalib.shape
    num_segments = kscalib_shape[-2]
    num_kx = kscalib_shape[-1]

    kscalib = np.reshape(kscalib, [-1, num_segments, num_kx])
    imcalib = fftc.ifftc(kscalib, axis=-1)
    imcalib_ref = np.conj(imcalib) * imcalib[:, :1, :]
    p1_calc = np.angle(np.mean(imcalib_ref[:, :, istart:iend]
                               * np.conj(imcalib_ref[:, :, (istart+1):(iend+1)]), axis=-1))
    p1_calc = np.expand_dims(p1_calc, axis=-1)

    x = np.arange(num_kx * 1.0)
    x = np.reshape(x, [1, 1, num_kx])
    imcalib_cor1 = imcalib_ref * np.exp(1j * x * p1_calc)
    p0_calc = np.angle(np.mean(imcalib_cor1, axis=-1))

    p1_calc = np.reshape(p1_calc, kscalib_shape[:-2] + p1_calc.shape[-2:])
    p0_calc = np.reshape(p0_calc, kscalib_shape[:-2] + p0_calc.shape[-1:] + (1,))

    return p0_calc, -p1_calc


def phase_correction(kspace, echo_train, kspace_fse_calib, echo_train_fse_calib):
    """Perform linear phase correction for FSE scans

    kspace: [phases, echoes, slices, channels, kz, ky, kx]
    echo_train: [phases, echoes, slices, 1, segments, 1]
    """
    x = np.arange(kspace.shape[-1] * 1.0)
    x = np.reshape(x, [1, 1, -1])

    p0, p1 = _compute_coefficients_ahncho(kspace_fse_calib)
    ksx = fftc.ifftc(kspace, axis=-1)

    num_phases = ksx.shape[0]
    num_echoes = ksx.shape[1]
    num_slices = ksx.shape[2]
    num_kz = ksx.shape[-3]

    for i_phase in range(num_phases):
        for i_echo in range(num_echoes):
            for i_slice in range(num_slices):
                for i_kz in range(num_kz):
                    ind = echo_train[i_phase, i_echo, i_slice, 0, i_kz, :, 0]
                    ind_calib = echo_train_fse_calib[i_phase, i_echo, i_slice, 0, :, 0]
                    ks_slice = ksx[i_phase, i_echo, i_slice, :, i_kz, :, :]

                    p0_slice = p0[i_phase, i_echo, i_slice, :, ind_calib, :]
                    p0_slice = np.transpose(p0_slice[ind, :, :], [1, 0, 2])

                    p1_slice = p1[i_phase, i_echo, i_slice, :, ind_calib, :]
                    p1_slice = np.transpose(p1_slice[ind, :, :], [1, 0, 2])

                    ks_slice *= np.exp(1j * (p0_slice + x * p1_slice))
                    ksx[i_phase, i_echo, i_slice, :, i_kz, :, :] = ks_slice

    kspace_cor = fftc.fftc(ksx, axis=-1)

    return kspace_cor
