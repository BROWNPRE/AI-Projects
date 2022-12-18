# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    # Tracks all of the visited states
    return depthFirstSearchHelper(problem, problem.getStartState(), list(), set(), None)

# A helper method that recursively runs DFS and returns a successful path is one is found, otherwise
# None will be returned.
def depthFirstSearchHelper(problem, state, path, visited, action):
    # Ignore visited states
    if state in visited:
        return None
    # Mark as visited
    visited.add(state)
    # Add action to path
    if action != None:
        path.append(action)
    # Reached a goal state
    if problem.isGoalState(state):
        return path

    # Search successor states for a path to goal state
    for next in problem.getSuccessors(state):
        if next[0] not in visited:
            nextPath = depthFirstSearchHelper(problem, next[0], path.copy(), visited, next[1])
            if nextPath != None:
                return nextPath

    # No paths to goal from this state
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Tracks all of the visited states
    visited = set()
    queue = util.Queue()
    # Insert start state into queue
    queue.push(((problem.getStartState(), None, None), list()))

    # NOTE: Queue contains (successor, path) tuples
    # Loop through queue until you find a goal state
    while not queue.isEmpty():
        popped = queue.pop()
        curState = popped[0]
        path = popped[1]
        # Ensure it isn't visited
        if curState[0] in visited:
            continue
        # Mark as visited
        visited.add(curState[0])
        # Add to path
        if curState[1] != None:
            path.append(curState[1])
        # Return if it's a goal state
        if problem.isGoalState(curState[0]):
            return path

        # Add successors to queue
        for next in problem.getSuccessors(curState[0]):
            if next[0] not in visited:
                queue.push((next, path.copy()))

    # Reached if the goal is unreachable from curState
    return None


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Tracks all of the visited states
    visited = set()
    queue = util.PriorityQueue()
    # Push start state
    queue.push(((problem.getStartState(), None, None), list(), 0), 0)
    # Queue contains (successor, path, totalCost) tuples

    # Loop through queue until you find a goal state
    while not queue.isEmpty():
        popped = queue.pop()
        curState = popped[0]
        path = popped[1]
        totalCost = popped[2]
        # Ensure it isn't visited
        if curState[0] in visited:
            continue
        # Mark as visited
        visited.add(curState[0])
        # Add to path
        if curState[1] != None:
            path.append(curState[1])
        # Return path if it's a goal state
        if problem.isGoalState(curState[0]):
            return path

        # Add successors to queue
        for next in problem.getSuccessors(curState[0]):
            if next[0] not in visited:
                newCost = totalCost + next[2]
                queue.push((next, path.copy(), newCost), newCost)

    # Reached if the goal is unreachable from curState
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Tracks all of the visited states
    visited = set()
    queue = util.PriorityQueue()
    # Push start state
    queue.push(((problem.getStartState(), None, None), list()), 0)
    # Queue contains (successor, path, dist_from_start) tuples

    # Loop through queue until you find a goal state
    while not queue.isEmpty():
        popped = queue.pop()
        curState = popped[0]
        path = popped[1]
        # Ensure it isn't visited
        if curState[0] in visited:
            continue
        # Mark as visited
        visited.add(curState[0])
        # Return path if it's a goal state
        if problem.isGoalState(curState[0]):
            return path

        # Add successors to queue
        for next in problem.getSuccessors(curState[0]):
            if next[0] not in visited:
                nextPath = path.copy()
                nextPath.append(next[1])
                fCost = problem.getCostOfActions(nextPath) + heuristic(next[0], problem)
                queue.push((next, nextPath), fCost)

    # Reached if the goal is unreachable from curState
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
