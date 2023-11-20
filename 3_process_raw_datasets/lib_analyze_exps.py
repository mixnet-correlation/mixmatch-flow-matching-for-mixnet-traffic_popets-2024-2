import numpy as np
from glob import glob
from io import TextIOWrapper
from os import walk, makedirs
from pandas import to_datetime
from os.path import join, dirname
from json import JSONEncoder, load as json_load, dump as json_dump



class NumPyJSONEncoder(JSONEncoder):
    """
    Custom JSONEncoder that is capable to encode NumPy objects.
    """
    
    def default(self, obj):
        
        if isinstance(obj, np.integer):
            return int(obj)
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return JSONEncoder.default(self, obj)



def walk_one_level(path: str, exclude_hidden: bool):
    
    subfolders = list()
    
    for _, dirs, _ in walk(path):
        
        # Join path and the identified subfolder.
        for d in dirs:
            
            if exclude_hidden and d.startswith("."):
                continue
            
            subfolders.append(join(path, d))
        
        # Set the dirs list's contents to nothing such that
        # the walk function will terminate after this level.
        del dirs[:]
    
    return sorted(subfolders)



def extract_nym_identities(
    initiator_file: str,
    responder_file: str):
    """
    Expect one file path each for the initiating and the responding Nym client
    of this experiment run. From the content of these two files, extract the Nym
    identity key part (that we use as identifier), and return both.
    """
    
    # Read full initiator address file into memory.
    with open(initiator_file, "r", encoding = "utf-8") as initiator_fp:
        initiator_addr_full = initiator_fp.read()
    
    assert "." in initiator_addr_full
    assert "@" in initiator_addr_full
    
    # Extract the Nym identity part.
    initiator_addr = initiator_addr_full.split(".")[0]
    
    assert len(initiator_addr) > 0

    # Read full responder address file into memory.
    with open(responder_file, "r", encoding = "utf-8") as responder_fp:
        responder_addr_full = responder_fp.read()
    
    assert "." in responder_addr_full
    assert "@" in responder_addr_full
    
    # Extract the Nym identity part.
    responder_addr = responder_addr_full.split(".")[0]
    
    assert len(responder_addr) > 0
    
    return initiator_addr, responder_addr



def convert_to_unix_nano(ts: str):
    """
    Take a timestamp formatted close to ISO 8601 and convert it to a UNIX
    timestamp with nanoseconds precision.
    """
    
    orig = to_datetime(ts, format = "%Y-%m-%dT%H:%M:%S.%fZ", exact = True)
    
    orig_seconds = f"{orig.timestamp()}".split(".")[0]
    orig_nanoseconds = f"{orig.microsecond:06}{orig.nanosecond:03}"
    orig_unix_str = f"{orig_seconds}{orig_nanoseconds}"
    
    orig_unix = np.uint64(orig_unix_str)
    
    assert orig.nanosecond == 0, f"{orig.nanosecond=} != 0"
    assert len(orig_seconds) == 10, f"{len(orig_seconds)=} != 10"
    assert len(orig_nanoseconds) == 9, f"{len(orig_nanoseconds)=} != 9"
    assert len(orig_unix_str) == 19, f"{len(orig_unix_str)=} != 19"
    assert orig_unix > 0, f"{orig_unix=} <= 0"
    assert str(orig_unix) == orig_unix_str, f"{str(orig_unix)=} != {orig_unix_str=}"
    
    # Parse crafted UNIX timestamp back to Pandas' Timestamp representation.
    ctrl = to_datetime(orig_unix, unit = "ns", origin = "unix")
    
    # Ensure we see exactly the same point in time as was passed into this function.
    assert ctrl.year == orig.year, f"{ctrl.year=} != {orig.year=}"
    assert ctrl.month == orig.month, f"{ctrl.month=} != {orig.month=}"
    assert ctrl.day == orig.day, f"{ctrl.day=} != {orig.day=}"
    assert ctrl.hour == orig.hour, f"{ctrl.hour=} != {orig.hour=}"
    assert ctrl.minute == orig.minute, f"{ctrl.minute=} != {orig.minute=}"
    assert ctrl.second == orig.second, f"{ctrl.second=} != {orig.second=}"
    assert ctrl.microsecond == orig.microsecond, f"{ctrl.microsecond=} != {orig.microsecond=}"
    assert ctrl.nanosecond == orig.nanosecond, f"{ctrl.nanosecond=} != {orig.nanosecond=}"
    
    ctrl_microseconds = f"{ctrl.microsecond:06}"
    ctrl_ts = f"{ctrl.strftime('%Y-%m-%dT%H:%M:%S')}.{ctrl_microseconds[:3]}Z"
    
    assert ctrl_ts == ts, f"{ctrl_ts=} != {ts=}"
    
    return orig_unix



def load_and_crop_flowpair_part(
    flowpair_part_file: str,
    start: np.uint64,
    end: np.uint64):
    """
    Load CSV-formatted 1/4th of a flowpair from 'flowpair_part_file' and filter out
    any sample for which its 'msg_timestamp_nanos' is strictly outside [start, end].
    """
    
    # Load part of flowpair from file and parse correctly into NumPy array of tuples.
    flowpair_part_raw = np.genfromtxt(
        flowpair_part_file,
        dtype = [np.uint64, np.int64],
        delimiter = ",",
        names = True,
        loose = False)
    
    # Below Boolean array indexing selects all rows from flowpair_part_raw where:
    #   1) 'msg_timestamp_nanos' is at least equal to parameter 'start'
    #    AND
    #   2) 'msg_timestamp_nanos' is at most equal to parameter 'end'.
    flowpair_part = flowpair_part_raw[((flowpair_part_raw["msg_timestamp_nanos"] >= start) & (flowpair_part_raw["msg_timestamp_nanos"] <= end))]
    
    assert np.all(np.logical_and((flowpair_part["msg_timestamp_nanos"] >= start), (flowpair_part["msg_timestamp_nanos"] <= end))) == True, \
        f"At least one element of flowpair_part['msg_timestamp_nanos'] lies outside [{start=}, {end=}]"
    
    return flowpair_part



def flip_sign(event: np.array):
    """
    Flip the sign of the 'msg_size_bytes' value of this event,
    turning it from a positive integer to a negative one.
    """
    
    assert event["msg_size_bytes"] > 0
    
    event["msg_size_bytes"] = -event["msg_size_bytes"]
    
    assert event["msg_size_bytes"] < 0
    
    return event



def merge_sort_convert_flowpair_half(
    to_gateway: np.array,
    from_gateway: np.array,
    run_start_ts: np.uint64):
    """
    Take the two parts of a flowpair half (i.e., the two channels of one flowpair
    endpoint), merge them, sort them in ascending timestamp order, and finally
    convert the UNIX nanoseconds timestamp to a timestamp in seconds relative to
    the first event in this half. Also add each event's relative timestamp to
    run_start_ts as third column.
    """

    # Merge the two parts of one half of a flowpair into a single array.
    half_unconverted_unsorted = np.concatenate((to_gateway, from_gateway))
    
    assert len(half_unconverted_unsorted) == len(to_gateway) + len(from_gateway), \
        f"{len(half_unconverted_unsorted)=} != {len(to_gateway)=} + {len(from_gateway)=}"
    
    # Sort by ascending UNIX nanoseconds timestamp.
    half_unconverted = np.sort(half_unconverted_unsorted, order = "msg_timestamp_nanos")
    
    assert np.all(np.diff(half_unconverted["msg_timestamp_nanos"]) > 0) == True, \
        f"{np.diff(half_unconverted['msg_timestamp_nanos'])=} <= 0"
    
    # Store timestamp of first event in this half for later deduction from
    # each subsequent timestamp (to turn them relative to this one).
    first_event_ts = half_unconverted[0]["msg_timestamp_nanos"]
    
    # Prepare a NumPy array of same size as 'half_unconverted' but with a float
    # type for the timestamps. We'll store the final flowpair half into this.
    half = np.zeros((len(half_unconverted), ), dtype = [
        ("msg_timestamp_relative_seconds", np.float64),
        ("msg_size_bytes", np.int64),
        ("msg_timestamp_relative_run_start_seconds", np.float64),
    ])
    
    test_val = (np.uint64(1670404923308623489) - np.uint64(1670404923308623488)) / np.float64(1000000000.0)
    assert test_val == 0.000000001
    assert type(test_val) == np.float64
    
    assert type(first_event_ts) == np.uint64
    assert type(run_start_ts) == np.uint64
    
    # Convert UNIX nanoseconds timestamp to relative timestamp from first packet in seconds.
    # Also add as third column the relative timestamp from this run's 'start' time in seconds.
    for idx, _ in enumerate(half_unconverted):
        
        assert type(half_unconverted[idx]["msg_timestamp_nanos"]) == np.uint64
        
        half[idx]["msg_timestamp_relative_seconds"] = \
            (half_unconverted[idx]["msg_timestamp_nanos"] - first_event_ts) / np.float64(1000000000.0)
        half[idx]["msg_size_bytes"] = \
            half_unconverted[idx]["msg_size_bytes"]
        half[idx]["msg_timestamp_relative_run_start_seconds"] = \
            (half_unconverted[idx]["msg_timestamp_nanos"] - run_start_ts) / np.float64(1000000000.0)
    
    assert half[0]["msg_timestamp_relative_seconds"] == 0.0, \
        f"{run_start_ts=}  =>  {half[0]['msg_timestamp_relative_seconds']=} != 0.0"
    assert np.all(half[1:]["msg_timestamp_relative_seconds"] > 0.0) == True, \
        f"{run_start_ts=}  =>  {half[1:]['msg_timestamp_relative_seconds']=} <= 0.0"
    assert np.all(half["msg_size_bytes"] != 0) == True, \
        f"{run_start_ts=}  =>  {half['msg_size_bytes']=} == 0"
    assert half[0]["msg_timestamp_relative_run_start_seconds"] >= 0.0, \
        f"{run_start_ts=}  =>  {half[0]['msg_timestamp_relative_run_start_seconds']=} < 0.0"
    assert np.all(half[1:]["msg_timestamp_relative_run_start_seconds"] > 0.0) == True, \
        f"{run_start_ts=}  =>  {half[1:]['msg_timestamp_relative_run_start_seconds']=} <= 0.0"
    
    return half



def has_at_least_one_stored_log_in_exp_time(
    gw_log: str,
    start: np.uint64,
    end: np.uint64,
    endpoint_addr: str):
    """
    Return true if the gateway's log file says that at least one message towards this endpoint
    was stored on-disk, because the endpoint was offline during the time of the experiment.
    If we find any such log line, we need to exclude this otherwise succeeded run from further
    consideration.
    The log line we're looking for is:
        [MIXCORR] [<ENDPOINT_IDENTITY>] WARNING Storing message on-disk because endpoint was unreachable
    """
    
    timestamps_raw = list()
    
    # The log line we'll be searching the gateway's log for, tailored to this endpoint's address.
    search_str = f"[MIXCORR] [{endpoint_addr}] WARNING Storing message on-disk because endpoint was unreachable"
    
    with open(gw_log, "r", encoding = "utf-8") as gw_fp:
        
        for line in gw_fp:
            
            # Find all relevant log lines and extract only each matching log line's timestamp.
            if search_str in line:
                timestamp = line.strip().split(" ")[0]
                timestamps_raw.append(timestamp)
    
    # If no such log line exists, there is no problem, thus return False.
    if len(timestamps_raw) == 0:
        return False

    # Convert each log line's timestamp to a UNIX timestamp with nanoseconds precision,
    # so that we can easily compare to the supplied 'start' and 'end' parameters.
    # Sort list in ascending numerical order to ensure our bounds checks work.
    timestamps = sorted(list(map(convert_to_unix_nano, timestamps_raw)))
    
    assert len(timestamps) == len(timestamps_raw), f"{len(timestamps)=} != {len(timestamps_raw)=}"
    
    # First check: If the unwanted log lines _start_ after the run has _ended_,
    # no overlap with the period we care about exists (all log lines occur strictly
    # later than the run interval). Thus, return False.
    if timestamps[0] > end:
        return False
    
    # Second check: If the unwanted log lines start before the end of the run but also
    # _end_ completely before the run has _started_, again no overlap exists (all log
    # lines occur strictly earlier than the run interval). Thus, return False.
    if timestamps[-1] < start:
        return False
    
    # Otherwise, we have some overlap and need to flag this run. Make sure this is visible
    # in the logs below, and return True.
    
    print(f"! ! !  WARNING: Run interval intersects with log lines at gateway indicating that messages were stored on disk:")
    print(f"! ! !  => Nym address of endpoint: {endpoint_addr}")
    print(f"! ! !  =>                   start: {start}")
    print(f"! ! !  =>                     end: {end}")
    print(f"! ! !  =>              timestamps: {timestamps}")
    
    return True



def check_expected_number_of_messages(flow_file: str, role: str):
    """
    Only exp02: Checks that we see the expected number of messages in 'flow_file' for a particular 'role' of:
    'initiator_to_gateway', 'initiator_from_gateway', 'responder_to_gateway', or 'responder_from_gateway'.
    """
    
    msgs_num = 0
    msgs_cnts_2413 = 0
    msgs_cnts_1675 = 0
    msgs_cnts_53 = 0
    
    with open(flow_file, "r", encoding = "utf-8") as flow_fp:
        
        for line in flow_fp:
            
            # Line indicating a message at origin.
            if line.endswith(",2413\n"):
                msgs_cnts_2413 += 1
            
            # Line indicating the payload of a message at destination.
            elif line.endswith(",1675\n"):
                msgs_cnts_1675 += 1
            
            # Line indicating the ACK of a message at destination.
            elif line.endswith(",53\n"):
                msgs_cnts_53 += 1
            
            msgs_num += 1
    
    # Ensure we see the correct number of the various messages
    # outlined above at the right roles of a flowpair.
    
    if role == "initiator_to_gateway":
        assert msgs_cnts_2413 == 3, \
            f"{msgs_cnts_2413=} != 3"
        assert (msgs_cnts_2413 + 1) == msgs_num, \
            f"({msgs_cnts_2413=} + 1) != {msgs_num=}"
        
    elif role == "initiator_from_gateway":
        assert msgs_cnts_1675 == 656, \
            f"{msgs_cnts_1675=} != 656"
        assert msgs_cnts_53 == 3, \
            f"{msgs_cnts_53=} != 3"
        assert (msgs_cnts_1675 + msgs_cnts_53 + 1) == msgs_num, \
            f"({msgs_cnts_1675=} + {msgs_cnts_53=} + 1) != {msgs_num=}"
        
    elif role == "responder_to_gateway":
        assert msgs_cnts_2413 == 656, \
            f"{msgs_cnts_2413=} != 656"
        assert (msgs_cnts_2413 + 1) == msgs_num, \
            f"({msgs_cnts_2413=} + 1) != {msgs_num=}"
        
    elif role == "responder_from_gateway":
        assert msgs_cnts_1675 == 3, \
            f"{msgs_cnts_1675=} != 3"
        assert msgs_cnts_53 == 656, \
            f"{msgs_cnts_53=} != 656"
        assert (msgs_cnts_1675 + msgs_cnts_53 + 1) == msgs_num, \
            f"({msgs_cnts_1675=} + {msgs_cnts_53=} + 1) != {msgs_num=}"



def create_processed_flowpair(
    proc_dir_our: str,
    proc_dir_dcoffea: str,
    sphinxflows_dir: str,
    initiator: str,
    responder: str,
    start: np.uint64,
    end: np.uint64):
    """
    For one flowpair between 'initiator' and 'responder', create a results folder in the processed dataset
    file system location and place (our format) a 'flowpair.json' file or (DeepCoFFEA's format) a
    'flowpair_initiator.json' and a 'flowpair_responder.json' file in there that contain the SphinxFlow
    contents of the respective four gateway log files cropped to [start, end].
    """
    
    initiator_to_gateway_file = glob(join(sphinxflows_dir, f"{initiator}_endpoint-to-gateway.sphinxflow"))
    initiator_from_gateway_file = glob(join(sphinxflows_dir, f"{initiator}_gateway-to-endpoint.sphinxflow"))
    responder_to_gateway_file = glob(join(sphinxflows_dir, f"{responder}_endpoint-to-gateway.sphinxflow"))
    responder_from_gateway_file = glob(join(sphinxflows_dir, f"{responder}_gateway-to-endpoint.sphinxflow"))
    
    assert len(initiator_to_gateway_file) == 1, f"{len(initiator_to_gateway_file)=} != 1"
    assert len(initiator_from_gateway_file) == 1, f"{len(initiator_from_gateway_file)=} != 1"
    assert len(responder_to_gateway_file) == 1, f"{len(responder_to_gateway_file)=} != 1"
    assert len(responder_from_gateway_file) == 1, f"{len(responder_from_gateway_file)=} != 1"
    
    # If this is exp02, make sure we see the expected number of messages on the different flow parts.
    # Mind that we perform these checks on the raw (i.e., uncropped) flow part version.
    if "exp02_nym-binaries-1.0.2_static-http-download_no-client-cover-traffic" in proc_dir_our:
        check_expected_number_of_messages(initiator_to_gateway_file[0], "initiator_to_gateway")
        check_expected_number_of_messages(initiator_from_gateway_file[0], "initiator_from_gateway")
        check_expected_number_of_messages(responder_to_gateway_file[0], "responder_to_gateway")
        check_expected_number_of_messages(responder_from_gateway_file[0], "responder_from_gateway")
    
    # Crop SphinxFlow samples for this run to interval [start, end].
    initiator_to_gateway = load_and_crop_flowpair_part(initiator_to_gateway_file[0], start, end)
    initiator_from_gateway = load_and_crop_flowpair_part(initiator_from_gateway_file[0], start, end)
    responder_to_gateway = load_and_crop_flowpair_part(responder_to_gateway_file[0], start, end)
    responder_from_gateway = load_and_crop_flowpair_part(responder_from_gateway_file[0], start, end)
    
    
    ## Our format.
    
    # Assemble final flowpair structure in our format.
    flowpair = {
        "start": start,
        "end": end,
        "initiator": {
            "nym_identity": initiator,
            "to_gateway": initiator_to_gateway,
            "from_gateway": initiator_from_gateway,
        },
        "responder": {
            "nym_identity": responder,
            "to_gateway": responder_to_gateway,
            "from_gateway": responder_from_gateway,
        },
    }
    
    # Create folder for this run at our format's dataset path.
    output_our_dir = join(proc_dir_our, f"{initiator}_{responder}")
    makedirs(output_our_dir, mode=0o755)
    
    # Write flowpair to JSON file using the above defined custom
    # JSONEncoder that is able to handle NumPy objects as well.
    
    flowpair_file = join(output_our_dir, "flowpair.json")
    
    with open(flowpair_file, "w", encoding = "utf-8") as flowpair_fp:
        json_dump(flowpair, flowpair_fp, cls = NumPyJSONEncoder)
        flowpair_fp.write("\n")
    
    
    ## DeepCoFFEA format.
    
    # Flip the signs of the message size entries of arrays from responder towards initiator,
    # i.e., the way back (responder to initiator) is represented as negative message sizes.
    responder_to_gateway_flipped = list(map(flip_sign, responder_to_gateway))
    initiator_from_gateway_flipped = list(map(flip_sign, initiator_from_gateway))
    
    # Merge the two parts of each flowpair half, sort them, and convert timestamps to relative.
    initiator_half = merge_sort_convert_flowpair_half(initiator_to_gateway, initiator_from_gateway_flipped, start)
    responder_half = merge_sort_convert_flowpair_half(responder_to_gateway_flipped, responder_from_gateway, start)
    
    # Create folder for this run at DeepCoFFEA's dataset path.
    output_dcoffea_dir = join(proc_dir_dcoffea, f"{initiator}_{responder}")
    makedirs(output_dcoffea_dir, mode=0o755)
    
    # Save flowpair halves to files.
    np.savetxt(join(output_dcoffea_dir, "flowpair_initiator"), initiator_half, fmt = ("%1.9f", "%1d", "%1.9f"), delimiter = "\t")
    np.savetxt(join(output_dcoffea_dir, "flowpair_responder"), responder_half, fmt = ("%1.9f", "%1d", "%1.9f"), delimiter = "\t")



def parse_run(
    proc_dir_our: str,
    proc_dir_dcoffea: str,
    idx: int,
    run_succeeded_file: str):
    """
    Parse the raw results files folder of a particular successful experiment run,
    signified via file path to its 'SUCCEEDED' file. Prepare all relevant file system
    locations for this run, load the experiment time and involved Nym identities,
    ensure that the experiment completed without unwanted log lines at the gateway,
    and finally extract the traffic flowpair cropped to the experiment time into the
    two formats (ours and DeepCoFFEA's) in their respective dataset folder.
    """
    
    run_dir = dirname(run_succeeded_file)
    
    if (idx % 250) == 0:
        print(f"[{idx:04}] Processing {run_dir}...")
    
    # These paths are specific to this experiment run.
    run_exp_file = join(run_dir, "experiment.json")
    initiator_addr_file = join(run_dir, "address_initiator_nym-socks5-client.txt")
    responder_addr_file = join(run_dir, "address_responder_nym-client.txt")
    
    # The responder address file is called differently in exp08.
    if "exp08_nym-binaries-v1.1.13_static-http-download" in proc_dir_our:
        responder_addr_file = join(run_dir, "address_responder_nym-network-requester.txt")
    
    # These paths are common to all runs of this specific experiment instantiation.
    exp_dir = dirname(run_dir)
    exp_gateway_log = join(exp_dir, "logs_docker-run_gateway-mixnodes.log")
    exp_sphinxflows_dir = join(exp_dir, "gateway_sphinxflows")
    
    # The gateway paths are different in exp08.
    if "exp08_nym-binaries-v1.1.13_static-http-download" in proc_dir_our:
        gw_dir = glob(join(dirname(dirname(run_dir)), "*_mixcorr-nym-gateway-ccx22-exp08_exp08_nym-binaries-v1.1.13_static-http-download"))[0]        
        exp_gateway_log = join(gw_dir, "logs_4-exp08-run-experiments-nym-gateway.log")
        exp_sphinxflows_dir = join(gw_dir, "gateway_sphinxflows")
    
    # Read JSON file containing the core experiment times [start, end] into memory.
    with open(run_exp_file, "r", encoding = "utf-8") as run_exp_fp:
        run_exp = json_load(run_exp_fp)
    
    assert run_exp["start"] > 0, f"{run_exp['start']=} <= 0"
    assert run_exp["end"] > 0, f"{run_exp['end']=} <= 0"
    assert len(str(run_exp["start"])) == 19, f"{len(str(run_exp['start']))=} != 19"
    assert len(str(run_exp["end"])) == 19, f"{len(str(run_exp['end']))=} != 19"
    assert run_exp["start"] < run_exp["end"], f"{run_exp['start']=} >= {run_exp['end']=}"
    
    # Convert start and end timestamps into NumPy data to be used for later calculations.
    start = np.uint64(run_exp["start"])
    end = np.uint64(run_exp["end"])
    
    assert start > 0, f"{start=} <= 0"
    assert start == run_exp["start"], f"{start=} != {run_exp['start']=}"
    assert end > 0, f"{end=} <= 0"
    assert end == run_exp["end"], f"{end=} != {run_exp['end']=}"
    
    # Read initiator and responder Nym identities into memory.
    initiator_addr, responder_addr = extract_nym_identities(initiator_addr_file, responder_addr_file)
    
    # Ensure that this successful run has not a single "WARNING Storing message
    # on-disk because endpoint was unreachable" log line at the gateway during
    # the time of the experiment ([start, end]).
    assert has_at_least_one_stored_log_in_exp_time(exp_gateway_log, start, end, initiator_addr) == False, \
        "At least one log message regarding on-disk stored messages for initiator during experiment time"
    assert has_at_least_one_stored_log_in_exp_time(exp_gateway_log, start, end, responder_addr) == False, \
        "At least one log message regarding on-disk stored messages for responder during experiment time"
    
    # Extract flowpair from raw into the two processed dataset locations.
    create_processed_flowpair(
        proc_dir_our,
        proc_dir_dcoffea,
        exp_sphinxflows_dir,
        initiator_addr,
        responder_addr,
        start,
        end)
    
    if (idx % 250) == 0:
        print()



def convert_timestamps(
    timestamps: list,
    start: np.uint64):
    """
    Takes a list of timestamp values and converts them to be relative to time specified
    as parameter start and in seconds instead of nanoseconds unit.
    """

    # Prepare an array of the exact size we'll need to convert each timestamp value.
    flow = np.zeros(len(timestamps), dtype = np.float64)

    for idx, timestamp_raw in enumerate(timestamps):
        
        # Make sure to convert each timestamp value to a np.uint64.
        timestamp = np.uint64(timestamp_raw)
        
        assert type(timestamp) == np.uint64, f"{type(timestamp)=} != np.uint64"
        assert timestamp > 0, f"{timestamp=} <= 0"

        # Convert each timestamp to be relative off of specified start time
        # and in seconds unit instead of the original nanoseconds unit.
        flow[idx] = (timestamp - start) / np.float64(1000000000.0)
    
    # The first converted timestamp _might_ be equal to 0.0,
    # all other ones _have_ to be greater than 0.0.
    assert flow[0] >= 0.0, f"{flow[0]=} < 0.0"
    assert np.all(flow[1:] > 0.0) == True, f"{flow[1:]=} <= 0.0"
    
    return flow



def write_flow_to_file(
    timestamps: np.array,
    flow_fp: TextIOWrapper):
    """
    Takes an array of timestamps and writes them out properly formatted to the supplied file handler.
    """
    
    # Convert each timestamp in the array to its properly formatted string representation.
    timestamps_str = [ f"{ts:.9f}" for ts in timestamps ]
    
    # Reduce the array of timestamp strings to one string, where each timestamp is
    # space-separated from the next. Add newline at the end. Write to file handler.
    flow_fp.write(f"{' '.join(timestamps_str)}\n" )
    
    # Make sure to flush buffers to disk before returning.
    flow_fp.flush()



def create_mixcorr_dataset(
    proc_dir_our: str,
    proc_dir_mixcorr: str):
    """
    Creates dataset in MixCorr format at `proc_dir_mixcorr` where we end up with 18 files
    for each dataset:
        { train, val, test } x { initiator, responder } x { to_gateway_data, from_gateway_data, from_gateway_ack }.
    The basis for this function is an already existing processed dataset in our format
    at `proc_dir_our`, thus function `parse_run()` above needs to have been run before
    this function can be run.
    We also require the dataset splitting to have been done already, thus the files:
        { flowpairs_train.json, flowpairs_val.json, flowpairs_test.json }
    need to exist at `proc_dir_our`.
    Messages `to_gateway_data` are signified as messages with size 2413 bytes,
    messages `from_gateway_data` are of size 1675 bytes, and messages `from_gateway_ack`
    are of size 53 bytes.
    """
    
    # We have three subsets for this dataset: TRAINING, VALIDATION, and TESTING.
    dataset_splits = [ "train", "val", "test" ]
    
    # Track the highest encountered relative timestamp in the TRAINING subset only.
    train_msg_time_max = np.float64(0.0)
    
    for split in dataset_splits:
        
        print(f"Considering flowpairs of dataset split '{split}' now...")
        
        # Open file descriptors to final output file paths for this split.
        initiator_to_gateway_data_fp = open(join(proc_dir_mixcorr, f"{split}_initiator_to_gateway_data"), "w", encoding = "utf-8")
        initiator_from_gateway_data_fp = open(join(proc_dir_mixcorr, f"{split}_initiator_from_gateway_data"), "w", encoding = "utf-8")
        initiator_from_gateway_ack_fp = open(join(proc_dir_mixcorr, f"{split}_initiator_from_gateway_ack"), "w", encoding = "utf-8")
        responder_to_gateway_data_fp = open(join(proc_dir_mixcorr, f"{split}_responder_to_gateway_data"), "w", encoding = "utf-8")
        responder_from_gateway_data_fp = open(join(proc_dir_mixcorr, f"{split}_responder_from_gateway_data"), "w", encoding = "utf-8")
        responder_from_gateway_ack_fp = open(join(proc_dir_mixcorr, f"{split}_responder_from_gateway_ack"), "w", encoding = "utf-8")

        # Read JSON file listing the flowpairs of this split.
        split_file = join(proc_dir_our, f"flowpairs_{split}.json")
        with open(split_file, "r", encoding = "utf-8") as split_fp:
            flowpair_dirs = json_load(split_fp)

        for flowpair_dir in flowpair_dirs:
            
            # Read flowpair JSON file.
            flowpair_file = join(proc_dir_our, flowpair_dir, "flowpair.json")
            with open(flowpair_file, "r", encoding = "utf-8") as flowpair_fp:
                flowpair = json_load(flowpair_fp)
            
            # Store run start timestamp in order to offset all timestamps from this.
            start = np.uint64(flowpair["start"])
            
            # Filter the flowpair values to the timestamp values of the respective endpoint role, direction, and message type.
            initiator_to_gateway_data = [ts for (ts, size) in flowpair["initiator"]["to_gateway"] if np.uint64(size) == 2413]
            initiator_from_gateway_data = [ts for (ts, size) in flowpair["initiator"]["from_gateway"] if np.uint64(size) == 1675]
            initiator_from_gateway_ack = [ts for (ts, size) in flowpair["initiator"]["from_gateway"] if np.uint64(size) == 53]
            responder_to_gateway_data = [ts for (ts, size) in flowpair["responder"]["to_gateway"] if np.uint64(size) == 2413]
            responder_from_gateway_data = [ts for (ts, size) in flowpair["responder"]["from_gateway"] if np.uint64(size) == 1675]
            responder_from_gateway_ack = [ts for (ts, size) in flowpair["responder"]["from_gateway"] if np.uint64(size) == 53]
            
            # Convert timestamps to be relative off of this run's start time.
            flow_initiator_to_gateway_data = convert_timestamps(initiator_to_gateway_data, start)
            flow_initiator_from_gateway_data = convert_timestamps(initiator_from_gateway_data, start)
            flow_initiator_from_gateway_ack = convert_timestamps(initiator_from_gateway_ack, start)
            flow_responder_to_gateway_data = convert_timestamps(responder_to_gateway_data, start)
            flow_responder_from_gateway_data = convert_timestamps(responder_from_gateway_data, start)
            flow_responder_from_gateway_ack = convert_timestamps(responder_from_gateway_ack, start)
            
            if split == "train":
                
                # If we are in the TRAINING dataset split, keep track of the highest seen relative timestamp value.
                
                if np.amax(flow_initiator_to_gateway_data) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_initiator_to_gateway_data)
                
                if np.amax(flow_initiator_from_gateway_data) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_initiator_from_gateway_data)
                
                if np.amax(flow_initiator_from_gateway_ack) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_initiator_from_gateway_ack)
                
                if np.amax(flow_responder_to_gateway_data) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_responder_to_gateway_data)
                
                if np.amax(flow_responder_from_gateway_data) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_responder_from_gateway_data)
                
                if np.amax(flow_responder_from_gateway_ack) > train_msg_time_max:
                    train_msg_time_max = np.amax(flow_responder_from_gateway_ack)
            
            # Write the extracted and converted message timestamps of this flowpair to the end of the respective file.
            write_flow_to_file(flow_initiator_to_gateway_data, initiator_to_gateway_data_fp)
            write_flow_to_file(flow_initiator_from_gateway_data, initiator_from_gateway_data_fp)
            write_flow_to_file(flow_initiator_from_gateway_ack, initiator_from_gateway_ack_fp)
            write_flow_to_file(flow_responder_to_gateway_data, responder_to_gateway_data_fp)
            write_flow_to_file(flow_responder_from_gateway_data, responder_from_gateway_data_fp)
            write_flow_to_file(flow_responder_from_gateway_ack, responder_from_gateway_ack_fp)
        
        # Make sure to close all file descriptors for this dataset split.
        initiator_to_gateway_data_fp.close()
        initiator_from_gateway_data_fp.close()
        initiator_from_gateway_ack_fp.close()
        responder_to_gateway_data_fp.close()
        responder_from_gateway_data_fp.close()
        responder_from_gateway_ack_fp.close()
        
        print(f"Done with flowpairs of dataset split '{split}'\n")
    
    # Write out highest seen relative timestamp value during TRAINING to dedicated file.
    train_msg_time_max_file = join(proc_dir_mixcorr, "train_msg_time_max")
    with open(train_msg_time_max_file, "w", encoding = "utf-8") as train_msg_time_max_fp:
        train_msg_time_max_fp.write(f"{train_msg_time_max:.9f}\n")