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


def user_to_dict(header):
    """Convert user parameters to dict."""
    user_dict = {}
    user_long = header['userParameters']['userParameterLong']
    user_double = header['userParameters']['userParameterDouble']
    user_string = header['userParameters']['userParameterString']

    for entry in user_long + user_double + user_string:
        user_dict[entry['name']] = entry['value']

    return user_dict


def dataset_to_cfl(dir_out, file_name, suffix="", dir_noise=None, verbose=False):
    """Convert ISMRMRD to CFL."""
    f = ismrmrd.Dataset(file_name, create_if_needed=False)
    header = xmltodict.parse(f.read_xml_header())['ismrmrdHeader']

    param_dict = user_to_dict(header)
    # is_scan_archive = param_dict['OrigFileFormat'] == "ScanArchive"
    num_kx = int(header['encoding']['encodedSpace']['matrixSize']['x'])
    num_ky = int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['maximum']) + 1
    num_kz = int(header['encoding']['encodingLimits']['kspace_encoding_step_2']['maximum']) + 1
    num_channels = int(header['acquisitionSystemInformation']['receiverChannels'])
    num_slices = int(header['encoding']['encodingLimits']['slice']['maximum']) + 1
    num_echoes = int(header['encoding']['encodingLimits']['contrast']['maximum']) + 1
    num_phases = int(header['encoding']['encodingLimits']['phase']['maximum']) + 1
    num_segments = int(header['encoding']['encodingLimits']['segment']['maximum']) + 1
    is_fse_with_calib = num_segments > 1
    chop_y = 1 - int(param_dict['ChopY'])
    chop_z = 1 - int(param_dict['ChopZ'])

    # Get noise file if it exists
    coil_config_uid = int(param_dict['CoilConfigUID'])
    file_search = "NoiseStatistics-Coil%d-*.h5" % coil_config_uid
    if dir_noise is not None:
        file_search = dir_noise + "/" + file_search
    file_list = glob.glob(file_search)
    if len(file_list):
        file_list = sorted(file_list)
        file_noise = file_list[-1]
        if verbose:
            print("Loading noise from file (%s)..." % file_noise)
        f_noise = h5py.File(file_noise, 'r')
        cplxcfg = h5py.get_config().complex_names
        h5py.get_config().complex_names = ('real','imag')
        opt_mat = np.copy(f_noise['Data']['OptimalTransformation'])
        noise = np.copy(f_noise['Data']['NoiseData'])
        h5py.get_config().complex_names = cplxcfg
        f_noise.close()

        prewhite = np.std(np.matmul(opt_mat.T, noise), axis=1)
        scale = np.mean(prewhite)
        opt_mat /= scale
        # For debugging
        if 0:
            print("Testing pre-whitening matrix...")
            print(prewhite)
            print(scale)
    else:
        if verbose:
            print("No noise file found...")
        rec_std = f.read_array('rec_std', 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        opt_mat = np.diag(rec_weight)
        # fn = open(os.path.join(dir_out, "NO_NOISE_STAT_FILE"), "w")
        # fn.close()

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
        num_acq = f.number_of_acquisitions()
    except:
        print("Unable to determine number of acquisitions! Empty?")
        return
    for i in wrap(range(num_acq)):
        acq = f.read_acquisition(i)
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
            if i_ky < num_ky:
                kspace[i_phase, i_echo, i_slice, :, i_kz, i_ky, :] = data
                if is_fse_with_calib:
                    echo_train[i_phase, i_echo, i_slice, 0, i_kz, i_ky, 0] = acq.idx.segment
            elif is_fse_with_calib:
                kspace_fse_cal[i_phase, i_echo, i_slice, :, i_ky - num_ky, :] = data
                echo_train_fse_cal[i_phase, i_echo, i_slice, 0, i_ky - num_ky, 0] = acq.idx.segment
    f.close()

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
            cfl.write(os.path.join(dir_out, "kspace_orig" + suffix), kspace)
        kspace = kspace_cor

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
    parser.add_argument("-n", "--noise", default=None,
                        help="directory for noise files (default:None)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose printing (default: False)")
    args = parser.parse_args()

    dataset_to_cfl(args.output, args.input, suffix=args.suffix,
                   dir_noise=args.noise, verbose=args.verbose)
