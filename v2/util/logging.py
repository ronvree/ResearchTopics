import os
import time
import pickle
import argparse


class Log:

    """
    Object for managing a log
    """

    # TODO -- option to print while logging?

    def __init__(self, args: argparse.Namespace):

        self._log_dir = args.log_directory
        self._logs = dict()

        self._print_msg = args.print_log

        if not os.path.isdir(self._log_dir):
            os.mkdir(self._log_dir)

    @staticmethod
    def build_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """

        :param parser:
        :return:
        """
        parser = parser or argparse.ArgumentParser('Log Arguments')

        parser.add_argument('--log_directory',
                            type=str,
                            default='./log',
                            help='The name of the directory in which a log should be built')
        parser.add_argument('--print_log',
                            action='set_true',
                            help='Log message entries are printed when this flag is set')

        return parser

    def log_message(self, message: str, log_name='log'):
        """
        Write a message to the log file
        :param message: the message string to be written to the log file
        :param log_name: the name of the log file
        """
        with open(self._log_dir + f'/{log_name}.txt', 'a') as f:
            timestamp = time.strftime('%d-%m-%Y %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')
        # Print the message if required
        if self._print_msg:
            print(message)

    def log_args(self, args: argparse.Namespace):
        """
        Log the parsed arguments
        :param args: argparse.Namespace object containing the parsed arguments
        """
        # Save the args in a text file
        with open(self._log_dir + '/args.txt', 'w') as f:
            for arg in vars(args):
                val = getattr(args, arg)
                if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                    val = f"'{val}'"
                f.write('{}: {}\n'.format(arg, val))
        # Pickle the args for possible reuse
        with open(self._log_dir + '/args.pickle', 'wb') as f:
            pickle.dump(args, f)

    def create_log(self, log_name: str, key_name: str, *value_names):  # TODO -- add to logs if it already exists!
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self._log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self._log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')



