from io import BytesIO, StringIO

import numpy as np
import pandas as pd


class File:
    """Class for handling lammps log files.

    Parameters
    ----------------------
    :param ifile: path to lammps log file
    :type ifile: string or file

    """

    def __init__(self, ifile):
        # Identifiers for places in the log file
        self.start_thermo_strings = [
            "Memory usage per processor",
            "Per MPI rank memory allocation"]
        self.stop_thermo_strings = ["Loop time", "ERROR", "Fix halt condition"]
        self.data_dict = {}
        self.keywords = []
        self.output_before_first_run = ""
        self.partial_logs = []
        self.intermittent_output = []
        self.simulation_settings = {}  # stores the most recent settings
        self.partial_simulation_settings = []  # stores the settings for each run
        if hasattr(ifile, "read"):
            self.logfile = ifile
        else:
            self.logfile = open(ifile, 'r')
        self.read_file_to_dict()
        self.parse_simulation_settings()

    def read_file_to_dict(self):
        contents = self.logfile.readlines()
        keyword_flag = False
        before_first_run_flag = True
        intermittent_output = False
        intermittent_output_tmp = []
        i = 0
        while i < len(contents):
            line = contents[i]
            if before_first_run_flag:
                self.output_before_first_run += line
            if intermittent_output:
                intermittent_output_tmp.append(line)

            if keyword_flag:
                keywords = line.split()
                tmpString = ""
                # Check wheter any of the thermo stop strings are in the
                # present line
                while not sum(
                        [string in line for string in self.stop_thermo_strings]) >= 1:
                    if "\n" in line:
                        tmpString += line
                    i += 1
                    if i < len(contents):
                        line = contents[i]
                    else:
                        break
                partialLogContents = pd.read_table(
                    StringIO(tmpString), sep=r'\s+')

                if (self.keywords != keywords):
                    # If the log keyword changes, i.e. the thermo data to be outputted changes,
                    # we flush all previous log data. This is a limitation of
                    # this implementation.
                    self.flush_dict_and_set_new_keyword(keywords)

                self.partial_dict = {}
                for name in keywords:
                    self.data_dict[name] = np.append(
                        self.data_dict[name], partialLogContents[name])
                    self.partial_dict[name] = np.append(
                        np.asarray([]), partialLogContents[name])
                self.partial_logs.append(self.partial_dict)
                keyword_flag = False

                self.intermittent_output.append(intermittent_output_tmp)
                intermittent_output_tmp = []
                intermittent_output = True

            # Check whether the string matches any of the start string
            # identifiers
            if sum([line.startswith(string)
                   for string in self.start_thermo_strings]) >= 1:
                keyword_flag = True
                before_first_run_flag = False
                intermittent_output = False
            i += 1
            # Remove first entry, which is empty
        self.intermittent_output = self.intermittent_output[1:]

    def parse_simulation_settings(self):
        """Read some simulation settings, such as `timestep`, `pair_style`, etc.,
          from the log file.

        A settings dict is created for each run. It can be accessed similar to the
        data dictionary, i.e. by using the :code:`get_simulation_settings(run_num
        )` method. The settings are stored in the :code:`partial_simulation_settings`.
        The latest settings are stored in the :code:`simulation_settings` attribute.

        Numeric value are cast to float if possible, otherwise they are stored as strings.
        """
        if not self.output_before_first_run:
            self.read_file_to_dict()
            if not self.output_before_first_run:
                raise ValueError(
                    "No output before first run found in log file.")
        sim_settings_kws = [
            'units',
            'timestep',
            'boundary',
            'atom_style',
            'pair_style',
            'bond_style',
            'angle_style',
            'dihedral_style',
            'improper_style']

        def parse_line(line):
            """Helper function to parse a line and extract key-value pairs."""
            settings = {}
            for kw in sim_settings_kws:
                if line.strip().startswith(kw):
                    line = line.split('#')[0]  # Remove comments
                    parts = [p.strip() for p in line.split()[1:]]
                    if len(parts) == 1:
                        try:
                            settings[kw] = float(parts[0])
                        except ValueError:
                            settings[kw] = parts[0]
                    else:
                        settings[kw] = ' '.join(parts)
            return settings

        for line in self.output_before_first_run.splitlines():
            line_settings = parse_line(line)
            self.simulation_settings.update(line_settings)
        self.partial_simulation_settings.append(
            self.simulation_settings.copy())

        for part in self.intermittent_output:
            # output contains only changes, so we need to start from
            # the previous settings and update accordingly
            tmp_settings = self.simulation_settings.copy()
            for line in part:
                line_settings = parse_line(line)
                tmp_settings.update(line_settings)

            self.partial_simulation_settings.append(tmp_settings)
            self.simulation_settings.update(tmp_settings)  # keep up to date

    def flush_dict_and_set_new_keyword(self, keywords):
        self.data_dict = {}
        for entry in keywords:
            self.data_dict[entry] = np.asarray([])
        self.keywords = keywords

    def get(self, entry_name, run_num=-1):
        """Get time-series from log file by name.

        Paramerers
        --------------------
        :param entry_name: Name of the entry, for example "Temp"
        :type entry_name: str
        :param run_num: Lammps simulations commonly involve several run-commands. Here you may choose what run you want the log data from. Default of :code:`-1` returns data from all runs concatenated
        :type run_num: int

        If the rows in the log file changes between runs, the logs are being flushed.
        """

        if run_num == -1:
            if entry_name in self.data_dict.keys():
                return self.data_dict[entry_name]
            else:
                return None
        else:
            if len(self.partial_logs) > run_num:
                partial_log = self.partial_logs[run_num]
                if entry_name in partial_log.keys():
                    return partial_log[entry_name]
                else:
                    return None
            else:
                return None

    def get_keywords(self, run_num=-1):
        """Return list of available data columns in the log file."""
        if run_num == -1:
            return sorted(self.keywords)
        else:
            if len(self.partial_logs) > run_num:
                return sorted(list(self.partial_logs[run_num].keys()))
            else:
                return None

    def to_exdir_group(self, name, exdirfile):
        group = exdirfile.require_group(name)
        for i, log in enumerate(self.partial_logs):
            subgroup = group.require_group(str(i))
            for key, value in log.items():
                key = key.replace("/", ".")
                subgroup.create_dataset(key, data=value)

    def to_dataframe(self, run_num=-1):
        return pd.DataFrame(self.partial_logs[run_num])

    def get_num_partial_logs(self):
        return len(self.partial_logs)

    @property
    def names(self):
        """Exposes the keywords returned by get_keywords."""
        return self.get_keywords()

    def get_simulation_settings(self, run_num=-1):
        """Get the simulation settings for a specific run."""
        return self.partial_simulation_settings[run_num]
