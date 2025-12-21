<div align="center">

  ![lichess-bot](https://github.com/lichess-bot-devs/lichess-bot-images/blob/main/lichess-bot-icon-400.png)

  <h1>lichess-bot</h1>

  A bridge between [lichess.org](https://lichess.org) and bots.
  <br>
  <strong>[Explore lichess-bot docs »](https://github.com/lichess-bot-devs/lichess-bot/wiki)</strong>
  <br>
  <br>
  [![Python Build](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml)
  [![Python Test](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml)
  [![Mypy](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml)

</div>

## Overview
Project Overview (Student Work)

This project implements a working Lichess chess chatbot using the Python lichess-bot framework.
The bot autonomously connects to Lichess, observes game events, and plays chess games against human players or other bots.

The move selection logic is AI-driven, combining classical game-search techniques with a machine learning model for position evaluation.

## AI-Based Player Algorithm
Decision-Making Strategy

The chatbot selects moves using an Iterative Deepening Alpha–Beta search algorithm.
For each legal move, the algorithm explores future positions up to a time-limited depth and selects the move with the highest expected outcome.
This approach allows the bot to:
Respect time constraints
Adapt search depth dynamically
Prune unpromising branches efficiently

## AI Model (Value Network)

To evaluate board positions during search, the bot uses a neural value model implemented in PyTorch.

Model characteristics:
Input: Encoded chess board representation (piece positions and side-to-move)
Architecture: Multi-layer perceptron (fully connected neural network)
Output: A scalar value in the range [-1, 1], representing how favorable a position is for the bot
Purpose: Replace hand-crafted heuristics with a learned evaluation function
The search algorithm uses this value to compare positions and guide move selection.

## Training Approach
The value model is trained offline on labeled chess positions.
Each training sample consists of:
A chess board position
A corresponding evaluation score (win/loss likelihood)
The training process minimizes prediction error using a regression loss function.
Once trained, the model is loaded by the bot and used during live games.

## Critical Analysis & Limitations

While the AI-driven approach improves decision quality, it has limitations:
The value network does not directly predict moves; it must be combined with a search algorithm.
The model depends on the quality and diversity of training data.
The board encoding is simplified and does not explicitly capture long-term strategic concepts such as plans or pawn structures.
Possible improvements include:
Richer board encodings (castling rights, repetition, move count)
Larger or convolutional neural networks
Combining value prediction with a policy (move prediction) network

## How This Project Meets the Evaluation Criteria

| Criterion                             | How it is satisfied                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Working chatbot using Lichess library | Bot connects to Lichess, accepts challenges, and plays games automatically     |
| AI-driven player algorithm            | Alpha–Beta search guided by a neural value model                               |
| Understanding & explanation           | Architecture, training, limitations, and improvements are explicitly described |


## Features
Supports:
- Every variant and time control
- UCI, XBoard, and Homemade engines
- Matchmaking (challenging other bots)
- Offering Draws and Resigning
- Participating in tournaments
- Accepting move takeback requests from opponents
- Saving games as PGN
- Local & Online Opening Books
- Local & Online Endgame Tablebases

Can run on:
- Python 3.9 and later
- Windows, Linux and MacOS
- Docker

## Steps
1. [Install lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install)
2. [Create a lichess OAuth token](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
3. [Setup the engine](https://github.com/lichess-bot-devs/lichess-bot/wiki/Setup-the-engine)
4. [Configure lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)
5. [Upgrade to a BOT account](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account)
6. [Run lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Run-lichess%E2%80%90bot)

## Advanced options
- [Create a homemade engine](https://github.com/lichess-bot-devs/lichess-bot/wiki/Create-a-homemade-engine)
- [Add extra customizations](https://github.com/lichess-bot-devs/lichess-bot/wiki/Extra-customizations)

<br />

## Acknowledgements
Thanks to the Lichess team, especially T. Alexander Lystad and Thibault Duplessis for working with the LeelaChessZero team to get this API up. Thanks to [Niklas Fiekas](https://github.com/niklasf) and his [python-chess](https://github.com/niklasf/python-chess) code which allows engine communication seamlessly.

## License
lichess-bot is licensed under the AGPLv3 (or any later version at your option). Check out the [LICENSE file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/LICENSE) for the full text.

## Citation
If this software has been used for research purposes, please cite it using the "Cite this repository" menu on the right sidebar. For more information, check the [CITATION file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/CITATION.cff).
