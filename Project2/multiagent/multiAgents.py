# multiAgents.py
# --------------
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

import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total = 0.0

        # Find pacman's distance from the closest ghost
        closestGhost = 9999999.0
        for ghostIdx in range(len(newGhostStates)):
            closestGhost = min(closestGhost, manhattanDistance(newGhostStates[ghostIdx].getPosition(), newPos))
        for ghostIdx in range(len(newGhostStates)):
            if closestGhost <= newScaredTimes[ghostIdx]:
                total += closestGhost
        if closestGhost < 3:
            total -= 10

        if newPos in currentGameState.getFood().asList():
            total += 5
        if currentGameState.hasWall(newPos[0], newPos[1]):
            total -= 2
        closestFood = []
        for cf in currentGameState.getFood().asList():
            closestFood.append(manhattanDistance(cf, newPos))
        total -= 0.05 * min(closestFood)

        return total

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    #def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        return self.minimax(gameState)

    def minimax(self, state):
        # Initial max values
        depth = 0
        agent = 0
        maxV = -999999
        maxAction = None
        # Search successors for action with highest value
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = self.value(sucState, depth + 1)
            # If a new max value was found
            if v > maxV:
                maxV = v
                maxAction = action
        return maxAction

    def value(self, state, depth):
        agent = depth % self.numAgents
        if depth / self.numAgents == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            return self.maxValue(state, agent, depth)
        else:
            return self.minValue(state, agent, depth)

    def maxValue(self, state, agent, depth):
        v = -999999
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = max(v, self.value(sucState, depth + 1))
        return v

    def minValue(self, state, agent, depth):
        v = 999999
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = min(v, self.value(sucState, depth + 1))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        return self.minimax(gameState)

    def minimax(self, state):
        # Initial max values
        depth = 0
        agent = 0
        alpha = -999999
        beta = 999999
        maxV = -999999
        maxAction = None
        # Search successors for action with highest value
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = self.value(sucState, depth + 1, alpha, beta)
            # If a new max value was found
            if v > maxV:
                maxV = v
                maxAction = action
            if maxV > beta:
                return maxV
            alpha = max(alpha, maxV)
        return maxAction

    def value(self, state, depth, alpha, beta):
        agent = depth % self.numAgents
        if depth / self.numAgents == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            return self.maxValue(state, agent, depth, alpha, beta)
        else:
            return self.minValue(state, agent, depth, alpha, beta)

    def maxValue(self, state, agent, depth, alpha, beta):
        v = -999999
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = max(v, self.value(sucState, depth + 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, state, agent, depth, alpha, beta):
        v = 999999
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = min(v, self.value(sucState, depth + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        return self.minimax(gameState)

    def minimax(self, state):
        # Initial max values
        depth = 0
        agent = 0
        alpha = -999999
        beta = 999999
        maxV = -999999
        maxAction = None
        # Search successors for action with highest value
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = self.value(sucState, depth + 1, alpha, beta)
            # If a new max value was found
            if v > maxV:
                maxV = v
                maxAction = action
        return maxAction

    def value(self, state, depth, alpha, beta):
        agent = depth % self.numAgents
        if depth / self.numAgents == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            return self.maxValue(state, agent, depth, alpha, beta)
        else:
            return self.expValue(state, agent, depth, alpha, beta)

    def maxValue(self, state, agent, depth, alpha, beta):
        v = -999999
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v = max(v, self.value(sucState, depth + 1, alpha, beta))
        return v

    def expValue(self, state, agent, depth, alpha, beta):
        v = 0
        p = 1.0 / len(state.getLegalActions(agent))
        for action in state.getLegalActions(agent):
            sucState = state.generateSuccessor(agent, action)
            v += p * self.value(sucState, depth + 1, alpha, beta)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    #newPos = successorGameState.getPacmanPosition()
    #newFood = successorGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodCount = len(currentGameState.getFood().asList())
    foodDist = 0
    if foodCount != 0:
        foodDist = aStarSearch(currentGameState.deepCopy(), currentGameState.getFood().asList())[1]

    capsuleCount = len(currentGameState.getCapsules())
    capsuleDist = 0
    if capsuleCount != 0:
        capsuleDist = aStarSearch(currentGameState.deepCopy(), currentGameState.getCapsules())[1]

    ghostPos = [ghostState.getPosition() for ghostState in currentGameState.getGhostStates()]
    truncGhostPos = list()
    for pos in ghostPos:
        truncGhostPos.append((math.trunc(pos[0]), math.trunc(pos[1])))
    closestGhost = aStarSearch(currentGameState.deepCopy(), truncGhostPos)
    ghostDist = closestGhost[1]
    ghostIdx = truncGhostPos.index(closestGhost[0])

    total = 0.0
    total -= 0.2 * foodDist
    if foodCount < 5:
        total += 10 - (2 * foodCount)

    total -= 10.0 * capsuleCount
    total -= 0.4 * capsuleDist

    total += currentGameState.getScore()
    if newScaredTimes[ghostIdx] > ghostDist:
        total += 15.0 / (ghostDist + 1)
    else:
        total -= 15.0 / (ghostDist + 1)

    if currentGameState.isWin():
        total += 5000
    if currentGameState.isLose():
        total -= 5000

    return total

def aStarHeuristic(pos, positions):
    # Return the manhattan distance to the food farthest away from pacman
    maxV = 999999
    for food in positions:
        maxV = min(maxV, util.manhattanDistance(pos, food))
    return maxV

def aStarSearch(state, positions, heuristic=aStarHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Tracks all of the visited states
    visited = dict()
    walls = state.getWalls()
    queue = util.PriorityQueue()
    # Push start state
    queue.push((state.getPacmanPosition(), 0), 0)
    # Queue contains (successor, path, dist_from_start) tuples

    # Loop through queue until you find a goal state
    while not queue.isEmpty():
        popped = queue.pop()
        pos = popped[0]
        count = popped[1]
        # Return path if it's a goal state
        if pos in positions:
            return (pos, count)

        visited[pos] = count
        # Add successors to queue
        for next in getAvailableMovements(state, pos):
            if next not in visited.keys():
                fCost = count + 1 + heuristic(pos, positions)
                queue.push((next, count + 1), fCost)

    # Reached if the goal is unreachable from curState
    return None

def getAvailableMovements(state, pos):
    availablePos = list()
    if not state.hasWall(pos[0] + 1, pos[1]):
        availablePos.append((pos[0] + 1, pos[1]))
    if not state.hasWall(pos[0], pos[1] + 1):
        availablePos.append((pos[0], pos[1] + 1))
    if not state.hasWall(pos[0] - 1, pos[1]):
        availablePos.append((pos[0] - 1, pos[1]))
    if not state.hasWall(pos[0], pos[1] - 1):
        availablePos.append((pos[0], pos[1] - 1))
    return availablePos


# Abbreviation
better = betterEvaluationFunction
