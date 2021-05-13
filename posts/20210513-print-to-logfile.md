# print to log file

```python
import sys
import os
import time

# Record the information printed in the terminal
class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    # 当前文件夹
    sys.stdout = Print_Logger(os.path.join('./', 'log_test.txt'))
    # 写入 'log_test.txt' 文件
    print("%s" % time.asctime())
```

