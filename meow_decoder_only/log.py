import os
from datetime import datetime
from inspect import currentframe, getframeinfo


class MeowLogger(object):
    def __init__(self):
        self.log_file = None

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()

    def __header(self, pid):
        now = datetime.now()
        frame_info = getframeinfo(currentframe().f_back.f_back)
        if pid:
            return "[\033[90m{}|\033[0m{}:{}|{}] ".format(
                now.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                os.path.basename(frame_info.filename),
                frame_info.lineno, os.getpid())
        return "[\033[90m{}|\033[0m{}:{}] ".format(
            now.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            os.path.basename(frame_info.filename), frame_info.lineno)

    def set_log_file(self, filename):
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = open(filename, "w")

    def log(self, content, muted=False):
        if muted:
            return
        if self.log_file is not None:
            self.log_file.write(content + "\n")
            self.log_file.flush()
            return
        print(content)

    def inf(self, line, pid=False, muted=False):
        self.log(self.__header(pid) + line, muted)

    def grey(self, line, pid=False, muted=False):
        self.log("\033[90m{}\033[0m".format(line), muted)



    def red(self, line, pid=False, muted=False):
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)


log = MeowLogger()
