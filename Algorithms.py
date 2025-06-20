import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict
from collections import deque

from zmq import QUEUE

class BFSGAgent():
    def __init__(self) -> None:
        pass
        #raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        print(1)
        q = deque([])
        used = set()
        prev = dict()
        q.append(env.get_initial_state())
        used.add(env.get_initial_state())
        while len(q) > 0:
          current = q.popleft()
          for action, t in sorted(env.succ(current).items()):
            if t[2]: 
              if env.is_final_state(t[0]):
                actions = [action]
                now = current
                cost = t[1]
                while now != env.get_initial_state():
                  prev_action, now, cost_here = prev[now]
                  print(actions)
                  actions.append(prev_action)
                  cost += cost_here
                actions.reverse()
                return (actions, cost, len(used) - len(q))
              else:
                continue
            else:
              if t[0] in used:
                continue
              q.append(t[0])
              used.add(t[0])
              prev[t[0]] = (action, current, t[1])
         
       # raise NotImplementedError
        


class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError   



class AStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

