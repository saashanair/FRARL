import os
import glob
import h5py
import warnings
import scipy.signal
import pandas as pd
import numpy as np

PATH_HIGHD = "/data/highD-dataset-v1.0/"

def get_straight_vehicles(df_meta):
    return df_meta[df_meta.numLaneChanges == 0].id

def savgol_filter(signal, polyorder=7):
    window_size = len(signal) // 2
    if window_size % 2 == 0:
        window_size -= 1
    if window_size <= polyorder:
        return None
    else:
        return scipy.signal.savgol_filter(signal, window_size, polyorder=polyorder)

def read_x_accelerations(track_fn, track_meta_fn):
    # read data frame from track and meta files
    df = pd.read_csv(track_fn, header=0)
    df_meta = pd.read_csv(track_meta_fn, header=0)

    # get all vehicles with no lane changes
    straight_veh_ids = get_straight_vehicles(df_meta)

    # gather all smoothed x accelerations
    x_accelerations = []
    init_vels = []
    max_length = 0
    for i, veh_id in enumerate(straight_veh_ids):
        print("{}/{}".format(i+1, len(straight_veh_ids)), end="\r")
        drivingDirection = df_meta[df_meta.id == veh_id].drivingDirection.values[0]
        a_x = df[df.id == veh_id].xAcceleration * (drivingDirection * 2 - 3)
        v_0 = df[df.id == veh_id].xVelocity.values[0] * (drivingDirection * 2 - 3)
        ax_smoothed = savgol_filter(a_x, polyorder=7)
        if ax_smoothed is None:
            warnings.warn('Signal length {} is too short of vehicle {}'.format(
                df_meta[df_meta.id == veh_id].numFrames.values[0], 
                veh_id))
            continue
        init_vels.append(v_0)
        x_accelerations.append(list(ax_smoothed))
        if len(ax_smoothed) > max_length:
            max_length = len(ax_smoothed)
            
    return x_accelerations, max_length, init_vels

def get_file_lists(path):
    listing = glob.glob(path)
    listing.sort()
    return listing

def write_to_h5_file(a_pos, v_pos, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('x_accelerations', data=a_pos)
    hf.create_dataset('initial_velocities', data=np.array(v_pos))
    hf.close()

def filter_out_neg_vel(x_accelerations, initial_velocities):
    v_pos = []
    a_pos = []
    v_neg = []
    a_neg = []
    delta_t = 0.04

    for i, x_accels in enumerate(x_accelerations):
        v = initial_velocities[i]       
        append_data = True
        for a in x_accels:
            v += a * delta_t
            if v < 0:
                append_data = False
                v_neg.append(initial_velocities[i])
                a_neg.append(x_accels)
                break
        if append_data:
            v_pos.append(initial_velocities[i])
            a_pos.append(x_accels)

    return a_pos, v_pos

def main():
    path_tracks = os.path.join(PATH_HIGHD, "data/*_tracks.csv")
    path_metas = os.path.join(PATH_HIGHD, "data/*_tracksMeta.csv")

    listing_tracks = get_file_lists(path_tracks)
    listing_metas = get_file_lists(path_metas)

    max_length = 0
    x_acc_fns = []
    init_vel_fns = []
    for fn_track, fn_meta in zip(listing_tracks, listing_metas):
        print(fn_track)
        x_acc_fn, max_length_fn, init_vel_fn = read_x_accelerations(fn_track, fn_meta)
        x_acc_fns += x_acc_fn
        init_vel_fns += init_vel_fn
        if max_length_fn > max_length:
            max_length = max_length_fn

    results = np.zeros((len(x_acc_fns), max_length))
    for i, x_acc in enumerate(x_acc_fns):
        results[i, :len(x_acc)] = x_acc
    print(results.shape)

    a_pos, v_pos = filter_out_neg_vel(results, np.array(init_vel_fns))

    print(np.array(a_pos).shape)
    write_to_h5_file(a_pos, v_pos, 'HighD_straight.h5')

if __name__ == "__main__":
    main()

