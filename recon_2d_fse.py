import numpy as np
import os
import glob
import h5py
import ismrmrd
import xmltodict
import argparse
from tqdm import tqdm
from fileio import cfl
from mrirecon import fse

def header_user_param_to_dict(header):
    """Convert user parameters to dict."""
    user_dict = {}
    user_long = list(header.userParameters.userParameterLong)
    user_double = list(header.userParameters.userParameterDouble)
    user_string = list(header.userParameters.userParameterString)
    user_base64 = list(header.userParameters.userParameterBase64)

    for entry in user_long + user_double + user_string + user_base64:
        user_dict[entry.name] = entry.value_

    return user_dict

def dataset_to_np(file_name, verbose=False):
    """Convert ISMRMRD to numpy array."""
    dataset = ismrmrd.Dataset(file_name, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    param_dict = header_user_param_to_dict(header)

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

    return kspace


def dataset_to_cfl(dir_out, file_name, suffix="", verbose=False):
    """Convert ISMRMRD to CFL."""
    kspace = dataset_to_np(file_name, verbose=verbose)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ISMRMRD files")
    parser.add_argument("input", action="store",
                        help="raw data in ISMRMRD format")
    parser.add_argument("-o", "--output", default="./",
                        help="output directory (default: ./)")
    parser.add_argument("-s", "--suffix", default="",
                        help="suffix to file output")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose printing (default: False)")
    args = parser.parse_args()

    dataset_to_cfl(args.output, args.input, suffix=args.suffix, verbose=args.verbose)
