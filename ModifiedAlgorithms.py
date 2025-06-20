import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict
from collections import deque

from zmq import QUEUE

#-----------------------------------------------------BFSG-------------------------------------
class BFSGAgent():
    def __init__(self) -> None:
        pass
        #raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        OPEN = deque([])
        CLOSED = set()
        prev = dict() # (state: (action, previous_state, cost))
        OPEN.append(env.get_initial_state())
        # while OPEN is not empty
        while len(OPEN) > 0:
          current = OPEN.popleft()
          CLOSED.add(current)

          # 0-down, 1-right, 2-up, 3-left
          for action, successor in sorted(env.succ(current).items()):
            (nextstate, cost, is_terminated) = successor

            # if GOAL: return (actions, cost, #nodes expanded)
            if is_terminated and env.is_final_state(nextstate):
                actions = [action]
                now = current
                while now != env.get_initial_state():
                    prev_action, now, cost_here = prev[now]
                    actions.append(prev_action)
                    cost += cost_here
                actions.reverse()
                return (actions, cost, len(CLOSED) )#- len(OPEN))

            else:
              if nextstate not in CLOSED and nextstate not in OPEN:
                # if hole, "expand" but dont add to OPEN:
                if is_terminated:
                  CLOSED.add(nextstate)
                # not hole:
                else:
                  OPEN.append(nextstate)
                prev[nextstate] = (action, current, cost)

#------------------------------   Agent, Node --------------------------------------------------

class Node():
    def __init__(self, state, prevNode=None, cost = 0,is_terminated=False):
        self.state = state
        self.prevNode = prevNode
        self.cost = cost
        self.is_terminated = is_terminated

class Agent():
    def __init__(self) -> None:
        self.path = dict()  # (Node : action)
        self.OPEN = heapdict.heapdict() # state: (cost, state,node)
        self.CLOSED = set() # set of States

    # input: final node, output: Tuple: (list of actions, total_cost)
    def solution(self, node: Node ) -> List[int]:
        actions = []
        cost = 0

        # while node is not initial state:
        while node.prevNode is not None:
            act = self.path[node] # action
            actions.append(act)
            # cost += node.cost
            # iterate to prev node
            node = node.prevNode
        actions.reverse()
        return actions

def h_manhattan(state, goal):
    N = (goal + 1) ** 0.5
    row =  state // N
    col = state % N
    return (N-1) - row + (N-1) - col

def h_haifa(env : HaifaEnv, state, Cpass = 100):
    minDistance = Cpass
    for goalState in env.get_goal_states():
        minDistance = min(minDistance,h_manhattan(state,goalState))
    return minDistance


class HNode(Node):
    def __init__(self,state, is_terminated, prevNode = None, cost = 0, fval = 0):
        super().__init__(state,prevNode,cost,is_terminated)
        self.fval = fval
        


#------------------------------   UCS --------------------------------------------------
class UCSAgent(Agent):
    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
      curNode = Node(env.get_initial_state(), None, 0,False)
      self.OPEN[curNode.state] = (curNode.cost,curNode.state,curNode)
      expanded_count = 0

      #while OPEN is not empty
      while len(self.OPEN) > 0:
        state, (cost, state,node) = self.OPEN.popitem() # gets "min" node from dictionary based on (g(),stateIndex)
        self.CLOSED.add(state)

        #if GOAL: return (actions, cost, #nodes expanded)
        if env.is_final_state(state):
            actions = self.solution(node)
            return (actions, cost, expanded_count) #except last goal 


            
        #node that was popped from OPEN and added to CLOSED
        # 0-down, 1-right, 2-up, 3-left
        expanded_count+=1
        
        if(node.is_terminated or node is None):
          continue
        else:
          for action, successor in sorted(env.succ(state).items()):
              (nextstate, transition_cost, is_terminated) = successor
              if(transition_cost is None or nextstate is None): continue
              newCost = cost + transition_cost
              child = Node(nextstate, node, newCost, is_terminated)

              # # if hole, "expand" but dont add to OPEN:
              # if is_terminated and not env.is_final_state(nextstate) and not nextstate in self.CLOSED:
              #     self.CLOSED.add(nextstate)# unnecessary

              if is_terminated and not env.is_final_state(nextstate):
                child.is_terminated = True

              # not hole:
              
              if nextstate not in self.OPEN and nextstate not in self.CLOSED:
                self.OPEN[nextstate] = (newCost, nextstate,child) #insertion
              elif nextstate in self.OPEN and newCost < self.OPEN[nextstate][0]:
                self.OPEN[nextstate] = (newCost, nextstate, child)#change of existing

              self.path[child] = action

#------------------------------   WA* --------------------------------------------------


class WeightedAStarAgent(Agent):

    def __init__(self):
        super().__init__()
        self.CLOSED = dict() # state: node

    def search(self, env: HaifaEnv, h_weight) -> Tuple[List[int], float, int]:
        curNode = HNode(env.get_initial_state(),False, None, 0,
                        h_haifa(env,env.get_initial_state()))
        self.OPEN[curNode.state] = (curNode.fval,curNode.state,curNode)
        expanded_count = 0

        while len(self.OPEN) > 0:
            state, (f, state, curNode) = self.OPEN.popitem()

            # cost returned from OPEN tuple is f value
            self.CLOSED[state] = curNode

            # if GOAL
            if env.is_final_state(state):
                actions = self.solution(curNode)
                return (actions, curNode.cost, expanded_count)

            expanded_count+=1
            # IF HOLE
            if curNode.is_terminated:
              # print("hole state: ",curNode.state)
              continue
            # print("successors:")
            for action, succesor in sorted(env.succ(state).items()):
                (next_state, transition_cost,is_terminated) = succesor
                if(next_state is None or transition_cost is None): continue
                new_g = curNode.cost + transition_cost
                new_f = (1-h_weight)*new_g + h_weight* h_haifa(env,next_state)

                # print("action",action,"next_State", next_state, f"terminated{is_terminated})")

                child = HNode(next_state,is_terminated, curNode, new_g, new_f)

                if next_state not in self.OPEN and next_state not in self.CLOSED:
                    self.OPEN[next_state] = (new_f, next_state, child)
                elif next_state in self.OPEN:
                    cost, state, cur_child = self.OPEN[next_state]
                    if child.fval < cur_child.fval:
                        self.OPEN[next_state] = (new_f, next_state, child)
                        
                else: # next_state in CLOSED
                    cur_child = self.CLOSED[next_state]
                    if child.fval < cur_child.fval:
                        self.OPEN[next_state] = (new_f, next_state, child)
                        del self.CLOSED[next_state]
                self.path[child] = action
#------------------------------   A* --------------------------------------------------

class AStarAgent(WeightedAStarAgent):
    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        return super().search(env, 0.5)

