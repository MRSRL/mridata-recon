"""Basic recon for 2D FSE datasets on mridata.org."""
import numpy as np
import os
import ismrmrd
import argparse
import imageio
from tqdm import tqdm
from fileio import cfl
from mrirecon import fse
from mrirecon import fftc

def isrmrmd_user_param_to_dict(header):
    """
    Store ISMRMRD header user parameters in a dictionary.

    Parameter
    ---------
    header : ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object

    Returns
    -------
    dict
        Dictionary containing custom user parameters
    """
    user_dict = {}
    user_long = list(header.userParameters.userParameterLong)
    user_double = list(header.userParameters.userParameterDouble)
    user_string = list(header.userParameters.userParameterString)
    user_base64 = list(header.userParameters.userParameterBase64)

    for entry in user_long + user_double + user_string + user_base64:
        user_dict[entry.name] = entry.value_

    return user_dict


def load_ismrmrd_to_np(file_name, verbose=False):
    """
    Load data from an ISMRMRD file to a numpy array.

    Raw data from the ISMRMRD file is loaded into a numpy array. If the ISMRMRD file includes the array 'rec_std' that contains the standard deviation of the noise, this information is used to pre-whiten the k-space data. If applicable, a basic phase correction is performed on the loaded k-space data.

    Parameters
    ----------
    file_name : str
        Name of ISMRMRD file
    verbose : bool, optional
        Turn on/off verbose print out

    Returns
    -------
    np.array
        k-space data in an np.array of dimensions [phase, echo, slice, coils, kz, ky, kx]
    ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    """
    dataset = ismrmrd.Dataset(file_name, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    param_dict = isrmrmd_user_param_to_dict(header)

    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    num_kz = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_echoes = header.encoding[0].encodingLimits.contrast.maximum + 1
    num_phases = header.encoding[0].encodingLimits.phase.maximum + 1
    num_segments = header.encoding[0].encodingLimits.segment.maximum + 1
    is_fse_with_calib = num_segments > 1

    chop_y = 1 - int(param_dict.get('ChopY', 1))
    chop_z = 1 - int(param_dict.get('ChopZ', 1))

    try:
        rec_std = dataset.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)

    if verbose:
        print("Data dims: (%d, %d, %d, %d, %d, %d, %d)" % (num_kx, num_ky, num_kz,
                                                           num_channels, num_slices,
                                                           num_echoes, num_phases))
    kspace = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                       num_kz, num_ky, num_kx], dtype=np.complex64)

    if is_fse_with_calib:
        echo_train = np.zeros([num_phases, num_echoes, num_slices, 1, num_kz, num_ky, 1],
                              dtype=np.uint)
        kspace_fse_cal = np.zeros([num_phases, num_echoes, num_slices, num_channels,
                                   num_segments, num_kx], dtype=np.complex64)
        echo_train_fse_cal = np.zeros([num_phases, num_echoes, num_slices, 1, num_segments, 1],
                                      dtype=np.uint)

    max_slice = 0
    wrap = lambda x: x
    if verbose:
        print("Loading data...")
        wrap = tqdm
    try:
        num_acq = dataset.number_of_acquisitions()
    except:
        print("Unable to determine number of acquisitions! Empty?")
        return
    for i in wrap(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1 # pylint: disable=E1101
        i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_echo = acq.idx.contrast           # pylint: disable=E1101
        i_phase = acq.idx.phase             # pylint: disable=E1101
        i_slice = acq.idx.slice             # pylint: disable=E1101
        if i_slice > max_slice:
            max_slice = i_slice
        sign = (-1) ** (i_ky * chop_y + i_kz * chop_z)
        data = np.matmul(opt_mat.T, acq.data) * sign
        if i_kz < num_kz:
            i_segment = acq.idx.segment # pylint: disable=E1101
            if i_ky < num_ky:
                kspace[i_phase, i_echo, i_slice, :, i_kz, i_ky, :] = data
                if is_fse_with_calib:
                    echo_train[i_phase, i_echo, i_slice, 0, i_kz, i_ky, 0] = i_segment
            elif is_fse_with_calib:
                kspace_fse_cal[i_phase, i_echo, i_slice, :, i_ky - num_ky, :] = data
                echo_train_fse_cal[i_phase, i_echo, i_slice, 0, i_ky - num_ky, 0] = i_segment
    dataset.close()

    max_slice += 1
    if num_slices != max_slice:
        if verbose:
            print("Actual number of slices different: %d/%d" % (max_slice, num_slices))
        kspace = kspace[:, :, :max_slice, :, :, :, :]
        if is_fse_with_calib:
            echo_train = echo_train[:, :, :max_slice, :, :, :, :]
            kspace_fse_cal = kspace_fse_cal[:, :, :max_slice, :, :, :]
            echo_train_fse_cal = echo_train_fse_cal[:, :, :max_slice, :, :, :]

    if is_fse_with_calib:
        if verbose:
            print("FSE phase correction...")
        if 0:
            print("writing files for debugging...")
            cfl.write("kspace", kspace)
            cfl.write("echo_train", echo_train)
            cfl.write("kspace_fse_cal", kspace_fse_cal)
            cfl.write("echo_train_fse_cal", echo_train_fse_cal)
        kspace_cor = fse.phase_correction(kspace, echo_train, kspace_fse_cal, echo_train_fse_cal)
        # for debugging
        if 0:
            cfl.write("kspace_orig" , kspace)
        kspace = kspace_cor

    return kspace, header


def transform(kspace, header, verbose=False):
    """
    Transform kspace data to image domain.

    If needed, cropping is performed based on information in the header.

    Parameters
    ----------
    kspace : np.array
        Data in k-space [phase, echo, slice, coils, z, ky, kx]
    header : ismrmrd.xsd.ismrmrdHeader
        ISMRMRD header object
    verbose : bool, optional
        Turn on/off print outs

    Returns
    -------
    np.array
        Data in image domain [phase, echo, slice, coils, z, y, x]
    """
    image = fftc.ifft2c(kspace)

    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    num_x = header.encoding[0].reconSpace.matrixSize.x
    num_y = header.encoding[0].reconSpace.matrixSize.y
    if num_x < num_kx:
        if verbose:
            print("Cropping data in x (%d to %d)..." % (num_kx, num_x))
        x0 = (num_kx - num_x) // 2
        x1 = x0 + num_x
        image = image[:, :, :, :, :, :, x0:x1]
    if num_y < num_ky:
        if verbose:
            print("Cropping data in y (%d to %d)..." % (num_ky, num_y))
        y0 = (num_ky - num_y) // 2
        y1 = y0 + num_y
        image = image[:, :, :, :, :, y0:y1, :]

    return image


def dataset_to_cfl(dir_out, file_name, suffix="", file_png=None, verbose=False):
    """
    Convert ISMRMRD to CFL files in specified directory.

    Parameters
    ----------
    dir_out : str
        Output directory to write CFL files
    file_name : str
        Name of ISMRMRD file
    suffix : str, optional
        Suffix to attach to output file names
    file_png : str, optional
        If not None, a png file will be written out
    verbose : bool, optional
        Turn on/off verbose print outs
    """
    kspace, header = load_ismrmrd_to_np(file_name, verbose=verbose)
    if verbose:
        print("Transforming k-space data to image domain...")
    image = transform(kspace, header, verbose=verbose)
    num_phases = kspace.shape[0]
    num_echoes = kspace.shape[1]

    if verbose:
        print("Writing files...")
    for i_phase in range(num_phases):
        for i_echo in range(num_echoes):
            suffix_i = suffix
            if num_echoes > 1:
                suffix_i = "_" + ("echo%02d" % i_echo) + suffix_i
            if num_phases > 1:
                suffix_i = "_" + ("phase%02d" % i_phase) + suffix_i

            cfl.write(os.path.join(dir_out, "kspace" + suffix_i),
                      kspace[i_phase, i_echo, :, :, :, :, :])
            cfl.write(os.path.join(dir_out, "image" + suffix_i),
                      image[i_phase, i_echo, :, :, :, :, :])

    if file_png is not None:
        if os.path.splitext(file_png)[1] != ".png":
            file_png += ".png"
        if verbose:
            print("Writing example png ({})...".format(file_png))
        energy = np.sum(np.abs(kspace) ** 2, axis=(-1, -2, -4))
        i_phase, i_echo, i_slice, i_z = np.where(energy == energy.max())
        image_out = image[i_phase[0], i_echo[0], i_slice[0], :, i_z[0], :, :]
        image_out = np.sqrt(np.sum(np.abs(image_out) ** 2, axis=0))
        image_out = image_out / np.max(image_out) * np.iinfo(np.uint8).max
        imageio.imwrite(file_png, image_out.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ISMRMRD files")
    parser.add_argument("input", action="store",
                        help="raw data in ISMRMRD format")
    parser.add_argument("-o", "--output", default="./",
                        help="output directory (default: ./)")
    parser.add_argument("-p", "--png", default=None,
                        help="png file name (default: None)")
    parser.add_argument("-s", "--suffix", default="",
                        help="suffix to file output")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose printing (default: False)")
    args = parser.parse_args()

    dataset_to_cfl(args.output, args.input, suffix=args.suffix,
                   file_png=args.png, verbose=args.verbose)
