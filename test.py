import gym
import numpy as np


def getChar():
    # figure out which function to use once, and store it in _func
    if "_func" not in getChar.__dict__:
        try:
            # for Windows-based systems
            import msvcrt # If successful, we are on Windows
            getChar._func=msvcrt.getch

        except ImportError:
            # for POSIX-based systems (with termios & tty support)
            import tty, sys, termios # raises ImportError if unsupported

            def _ttyRead():
                fd = sys.stdin.fileno()
                oldSettings = termios.tcgetattr(fd)

                try:
                    tty.setcbreak(fd)
                    answer = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, oldSettings)

                return answer

            getChar._func=_ttyRead

    return getChar._func()


env = gym.make('MontezumaRevenge-v0') # (210, 160, 3) array image
#env = gym.make('MontezumaRevenge-ram-v0') # RAM, only 128 bytes
env.reset()
env.render()
# action 0-17   a-r

for t in range(1000):
  action = ord(getChar()) - ord('a')
  print("action: ", action)
  nextstate, reward, is_terminal, debug_info = env.step(action)
  print(nextstate.shape, reward, is_terminal, debug_info)
  
  if t == 0:
  	np.save("image", nextstate)

  env.render()

exit()