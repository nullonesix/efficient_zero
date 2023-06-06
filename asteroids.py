
# Notes:
# random.randrange returns an int
# random.uniform returns a float
# p for pause
# j for toggle showing FPS
# o for frame advance whilst paused

import pygame
import sys
import os
import random
from pygame.locals import *
from util.vectorsprites import *
from ship import *
from stage import *
from badies import *
from shooter import *
from soundManager import *


class Asteroids():

    explodingTtl = -1

    def __init__(self):
        self.stage = Stage('Asteroids', (1024, 768))
        # self.stage = Stage('Asteroids', (1024/2, 768/2))
        self.paused = False
        self.showingFPS = False
        self.frameAdvance = False
        self.gameState = "attract_mode"
        # self.gameState = 'playing'
        self.rockList = []
        # self.createRocks(3)
        self.createRocks(10)
        self.saucer = None
        self.secondsCount = 1
        self.score = 0
        self.ship = None
        self.lives = 0

    def initialiseGame(self):
        self.gameState = 'playing'
        [self.stage.removeSprite(sprite)
         for sprite in self.rockList]  # clear old rocks
        if self.saucer is not None:
            self.killSaucer()
        self.startLives = 1 # 5
        self.createNewShip()
        self.createLivesList()
        self.score = 0
        self.rockList = []
        # self.numRocks = 3
        self.numRocks = 80
        self.nextLife = 10000

        self.createRocks(self.numRocks)
        self.secondsCount = 1

    def createNewShip(self):
        if self.ship:
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
        self.ship = Ship(self.stage)
        self.stage.addSprite(self.ship.thrustJet)
        self.stage.addSprite(self.ship)

    def createLivesList(self):
        self.lives += 1
        self.livesList = []
        for i in range(1, self.startLives):
            self.addLife(i)

    def addLife(self, lifeNumber):
        self.lives += 1
        ship = Ship(self.stage)
        self.stage.addSprite(ship)
        ship.position.x = self.stage.width - \
            (lifeNumber * ship.boundingRect.width) - 10
        ship.position.y = 0 + ship.boundingRect.height
        self.livesList.append(ship)

    def createRocks(self, numRocks):
        for _ in range(0, numRocks):
            position = Vector2d(random.randrange(-10, 10),
                                random.randrange(-10, 10))

            newRock = Rock(self.stage, position, Rock.largeRockType)
            self.stage.addSprite(newRock)
            self.rockList.append(newRock)

    def playGame(self, move):

        clock = pygame.time.Clock()

        frameCount = 0.0
        timePassed = 0.0
        self.fps = 0.0
        # Main loop
        # while True:
        if True:

            # calculate fps
            timePassed += clock.tick(60)
            frameCount += 1
            if frameCount % 10 == 0:  # every 10 frames
                # nearest integer
                self.fps = round((frameCount / (timePassed / 1000.0)))
                # reset counter
                timePassed = 0
                frameCount = 0

            self.secondsCount += 1

            self.input(move)

            # pause
            if self.paused and not self.frameAdvance:
                self.displayPaused()
                # continue
                return

            self.stage.screen.fill((10, 10, 10))
            self.stage.moveSprites()
            self.stage.drawSprites()
            self.doSaucerLogic()
            self.displayScore()
            if self.showingFPS:
                self.displayFps()  # for debug
            self.checkScore()

            # Process keys
            if self.gameState == 'playing':
                self.playing()
            elif self.gameState == 'exploding':
                self.exploding()
            else:
                self.displayText()

            # Double buffer draw
            pygame.display.flip()

    def playing(self):
        if self.lives == 0:
        # if False:
            self.gameState = 'attract_mode'
        else:
            self.processKeys()
            self.checkCollisions()
            if len(self.rockList) == 0:
                self.levelUp()

    def doSaucerLogic(self):
        if self.saucer is not None:
            if self.saucer.laps >= 2:
                self.killSaucer()

        # Create a saucer
        if self.secondsCount % 2000 == 0 and self.saucer is None:
            randVal = random.randrange(0, 10)
            if randVal <= 3:
                self.saucer = Saucer(
                    self.stage, Saucer.smallSaucerType, self.ship)
            else:
                self.saucer = Saucer(
                    self.stage, Saucer.largeSaucerType, self.ship)
            self.stage.addSprite(self.saucer)

    def exploding(self):
        self.explodingCount += 1
        if self.explodingCount > self.explodingTtl:
            self.gameState = 'playing'
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
            self.ship.shipDebrisList = []

            if self.lives == 0:
                self.ship.visible = False
            else:
                self.createNewShip()

    def levelUp(self):
        self.numRocks += 1
        self.createRocks(self.numRocks)

    # move this kack somewhere else!
    def displayText(self):
        # font1 = pygame.font.Font('../res/Hyperspace.otf', 50)
        # font2 = pygame.font.Font('../res/Hyperspace.otf', 20)
        # font3 = pygame.font.Font('../res/Hyperspace.otf', 30)

        font1 = pygame.font.SysFont(pygame.font.get_default_font(), 50)
        font2 = pygame.font.SysFont(pygame.font.get_default_font(), 20)
        font3 = pygame.font.SysFont(pygame.font.get_default_font(), 30)

        titleText = font1.render('Asteroids', True, (180, 180, 180))
        titleTextRect = titleText.get_rect(centerx=self.stage.width/2)
        titleTextRect.y = self.stage.height/2 - titleTextRect.height*2
        self.stage.screen.blit(titleText, titleTextRect)

        keysText = font2.render(
            '(C) 1979 - 2021 Boularbah Ismail.', True, (255, 255, 255))
        keysTextRect = keysText.get_rect(centerx=self.stage.width/2)
        keysTextRect.y = self.stage.height - keysTextRect.height - 20
        self.stage.screen.blit(keysText, keysTextRect)

        instructionText = font3.render(
            'Press enter to start', True, (200, 200, 200))
        instructionTextRect = instructionText.get_rect(
            centerx=self.stage.width/2)
        instructionTextRect.y = self.stage.height/2 - instructionTextRect.height
        self.stage.screen.blit(instructionText, instructionTextRect)

    def displayScore(self):
        # font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
        # font2 = pygame.font.Font('../res/Hyperspace.otf', 25)

        font1 = pygame.font.SysFont(pygame.font.get_default_font(), 30)
        font2 = pygame.font.SysFont(pygame.font.get_default_font(), 25)

        scoreStr = str("%02d" % self.score)
        scoreText = font2.render('Score:' + scoreStr, True, (200, 200, 200))
        scoreTextRect = scoreText.get_rect(centerx=100, centery=45)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def displayPaused(self):
        if self.paused:
            font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
            pausedText = font1.render("Paused", True, (255, 255, 255))
            textRect = pausedText.get_rect(
                centerx=self.stage.width/2, centery=self.stage.height/2)
            self.stage.screen.blit(pausedText, textRect)
            pygame.display.update()

    # Should move the ship controls into the ship class
    def input(self, events):
        self.frameAdvance = False
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                if self.gameState == 'playing':
                    if event.key == K_SPACE:
                        self.ship.fireBullet()
                    elif event.key == K_b:
                        self.ship.fireBullet()
                    elif event.key == K_h:
                        self.ship.enterHyperSpace()
                    elif event.key == K_LEFT:
                        self.ship.rotateLeft()
                    elif event.key == K_RIGHT:
                        self.ship.rotateRight()
                    if event.key == K_UP:
                        self.ship.increaseThrust()
                        self.ship.thrustJet.accelerating = True
                    else:
                        self.ship.thrustJet.accelerating = False

                elif self.gameState == 'attract_mode':
                    # Start a new game
                    if event.key == K_RETURN:
                        self.initialiseGame()

                if event.key == K_p:
                    if self.paused:  # (is True)
                        self.paused = False
                    else:
                        self.paused = True

                if event.key == K_j:
                    if self.showingFPS:  # (is True)
                        self.showingFPS = False
                    else:
                        self.showingFPS = True

                if event.key == K_f:
                    pygame.display.toggle_fullscreen()

                # if event.key == K_k:
                    # self.killShip()
            elif event.type == KEYUP:
                if event.key == K_o:
                    self.frameAdvance = True

    def processKeys(self):
        key = pygame.key.get_pressed()

        if key[K_LEFT] or key[K_z]:
            self.ship.rotateLeft()
        elif key[K_RIGHT] or key[K_x]:
            self.ship.rotateRight()

        if key[K_UP] or key[K_n]:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
        else:
            self.ship.thrustJet.accelerating = False

    # Check for ship hitting the rocks etc.

    def checkCollisions(self):

        # Ship bullet hit rock?
        newRocks = []
        shipHit, saucerHit = False, False

        # Rocks
        for rock in self.rockList:
            rockHit = False

            if not self.ship.inHyperSpace and rock.collidesWith(self.ship):
                p = rock.checkPolygonCollision(self.ship)
                if p is not None:
                    shipHit = True
                    rockHit = True

            if self.saucer is not None:
                if rock.collidesWith(self.saucer):
                    saucerHit = True
                    rockHit = True

                if self.saucer.bulletCollision(rock):
                    rockHit = True

                if self.ship.bulletCollision(self.saucer):
                    saucerHit = True
                    self.score += self.saucer.scoreValue

            if self.ship.bulletCollision(rock):
                rockHit = True

            if rockHit:
                self.rockList.remove(rock)
                self.stage.spriteList.remove(rock)

                if rock.rockType == Rock.largeRockType:
                    playSound("explode1")
                    newRockType = Rock.mediumRockType
                    self.score += 50
                elif rock.rockType == Rock.mediumRockType:
                    playSound("explode2")
                    newRockType = Rock.smallRockType
                    self.score += 100
                else:
                    playSound("explode3")
                    self.score += 200

                if rock.rockType != Rock.smallRockType:
                    # new rocks
                    for _ in range(0, 2):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)

                self.createDebris(rock)

        # Saucer bullets
        if self.saucer is not None:
            if not self.ship.inHyperSpace:
                if self.saucer.bulletCollision(self.ship):
                    shipHit = True

                if self.saucer.collidesWith(self.ship):
                    shipHit = True
                    saucerHit = True

            if saucerHit:
                self.createDebris(self.saucer)
                self.killSaucer()

        if shipHit:
            self.killShip()

            # comment in to pause on collision
            #self.paused = True

    def killShip(self):
        stopSound("thrust")
        playSound("explode2")
        self.explodingCount = 0
        self.lives -= 1
        if (self.livesList):
            ship = self.livesList.pop()
            self.stage.removeSprite(ship)

        self.stage.removeSprite(self.ship)
        self.stage.removeSprite(self.ship.thrustJet)
        self.gameState = 'exploding'
        self.ship.explode()

    def killSaucer(self):
        stopSound("lsaucer")
        stopSound("ssaucer")
        playSound("explode2")
        self.stage.removeSprite(self.saucer)
        self.saucer = None

    def createDebris(self, sprite):
        for _ in range(0, 25):
            position = Vector2d(sprite.position.x, sprite.position.y)
            debris = Debris(position, self.stage)
            self.stage.addSprite(debris)

    def displayFps(self):
        font2 = pygame.font.Font('../res/Hyperspace.otf', 15)
        fpsStr = str(self.fps)+(' FPS')
        scoreText = font2.render(fpsStr, True, (255, 255, 255))
        scoreTextRect = scoreText.get_rect(
            centerx=(self.stage.width/2), centery=15)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def checkScore(self):
        if self.score > 0 and self.score > self.nextLife:
            playSound("extralife")
            self.nextLife += 10000
            self.addLife(self.lives)


# Script to run the game
if not pygame.font:
    print('Warning, fonts disabled')
if not pygame.mixer:
    print('Warning, sound disabled')



import numpy as np


from time import time
import copy
from pygame.locals import *
from pygame.event import Event

class Node():
    def __init__(self):
        # self.forward = None
        # self.left = None
        # self.right = None
        # self.fire = None
        self.moves = {0: None, 1: None, 2: None, 3: None}
        # self.n_wins = 0
        # self.n_total = 1
        # self.q = torch.Tensor([0.0]).to(device)
        # self.q = quality()
        self.q = None
        self.compressed = None
    
def quality_tree(move):
    if move == None:
        return 0.5 # could put neural network here 
    return move.n_wins / move.n_total

def add_node(node):
    for key, value in node.moves.items():
        if node.moves[key] == None:
            new_node = Node()
            ai_move_h = np.zeros((1, len(ai_moves)))
            ai_move_h[0][key] = 1.0
            ai_move_h_t = torch.Tensor(ai_move_h).to(device)
            compressed_pa = torch.concatenate([node.compressed, ai_move_h_t], dim=1)
            compressed_pa_np1 = temporal(compressed_pa)
            new_node.compressed = compressed_pa_np1
            compressed_q = quality(compressed_pa_np1)
            new_node.q = compressed_q[0][0]
            node.moves[key] = new_node
            node.q = max(node.q, new_node.q)
            return node.q
    key = np.random.randint(4)
    return max(node.q, add_node(node.moves[key]))

            
def generate(compressed):
    root = Node()
    root.compressed = compressed
    root.q = quality(compressed)
    start_time = time()
    count = 0
    while True:
        if time() - start_time > 1.0:
            break
        # add_node_start_time = time()
        add_node(root)
        # print(time()-add_node_start_time)
        count += 1
    move = np.argmax([root.moves[key].q.detach().cpu() for key in ai_moves])
    print('0: left, 1: right, 2: space, 3: up')
    print([root.moves[key].q.detach().cpu() for key in ai_moves])
    print('nodes', count)
    return move, root.q

def action_to_move(move):
    if move == 'left':
        move = K_LEFT
    elif move == 'right':
        move = K_RIGHT
    elif move == 'fire':
        move = K_SPACE
    elif move == 'forward':
        move = K_UP
    # newevent = Event(KEYDOWN, key=move, mod=KMOD_NONE) #create the event
    # print(move)
    # return [newevent]    
    return move

# root = Node()
# generate()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

class Compress(nn.Module):

    def __init__(self):
        super(Compress, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 64, 64)
        self.conv2 = nn.Conv2d(1, 1, 8, 8) # torch.Size([1, 507949])
        # self.conv3 = nn.Conv2d(1, 1, 64, 64)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(120, 128)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x

class Temporal(nn.Module):

    def __init__(self):
        super(Temporal, self).__init__()
        # self.conv1 = nn.Conv1d()
        # self.conv1 = nn.Conv2d(1, 1, 8, 8)
        # self.conv2 = nn.Conv2d(1, 1, 8, 8)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 6)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x

class Quality(nn.Module):

    def __init__(self):
        super(Quality, self).__init__()
        # self.conv1 = nn.Conv1d()
        # self.conv1 = nn.Conv2d(1, 1, 8, 8)
        # self.conv2 = nn.Conv2d(1, 1, 8, 8)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 1)
        # self.fc3 = nn.Linear(6, 2)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        # output = F.log_softmax(x, dim=1)
        # output = F.relu(x)
        output = x
        # output = F.sigmoid(x)
        return output
        # return x

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(1, 1)
#         self.fc2 = nn.Linear(1, 1)

#     def forward(self, inputs):
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#         # x = x + inputs
#         return x

# device = torch.device("cuda")
# model = Net().to(device)
# opt = optim.AdamW(model.parameters())

up = pygame.event.Event(KEYDOWN, unicode="up", key=K_UP, mod=KMOD_NONE)
left = pygame.event.Event(KEYDOWN, unicode="left", key=K_LEFT, mod=KMOD_NONE)
right = pygame.event.Event(KEYDOWN, unicode="right", key=K_RIGHT, mod=KMOD_NONE)
space = pygame.event.Event(KEYDOWN, unicode="space", key=K_SPACE, mod=KMOD_NONE)

# ai_moves = [up, left, right, space]
ai_moves = [0, 1, 2, 3]
ai_moves_to_event = {0: left, 1: right, 2: space, 3: up}
# ai_moves = [up, left, right]

initSoundManager()
game = Asteroids()  # create object game from class Asteroids
start = pygame.event.Event(KEYDOWN, unicode="return", key=K_RETURN, mod=KMOD_NONE)
window = pygame.event.Event(KEYDOWN, unicode="window", key=K_f, mod=KMOD_NONE)
game.playGame([window])
# game.playGame([start])

# def get_axes():
#     # pixels = pygame.surfarray.pixels2d(self.stage.screen)
#     pixels = pygame.surfarray.pixels3d(game.stage.screen)
#     pixels = pixels[:,:,0]
#     print(pixels)
#     # print(pixels[:,:,0].shape)
#     # print(set(pixels.flatten()))
#     del pixels

# from tree import *

device = torch.device("cpu")
# device = torch.device("cuda:0")
compress = Compress().to(device)
temporal = Temporal().to(device)
quality = Quality().to(device)

# print(summary(compress, input_size=(1, 1920, 1080)))
# print(summary(temporal, input_size=(1, 10)))
# print(summary(quality, input_size=(1, 6)))

compress.load_state_dict(torch.load('compress.model'))
temporal.load_state_dict(torch.load('temporal.model'))
quality.load_state_dict(torch.load('quality.model'))
parameters = list(compress.parameters())+list(temporal.parameters())+list(quality.parameters())
opt = optim.Adam(parameters)
# opt = optim.Adam(list(compress.parameters())+list(quality.parameters()))
# opt_temporal = optim.Adam(temporal.parameters())
count = 0

pixels = pygame.surfarray.pixels3d(game.stage.screen)
pixels = pixels[:,:,0]
pixels = torch.from_numpy(pixels.astype(np.float32)).to(device)
pixels = torch.reshape(pixels, (1, 1920, 1080))
old_pixels = pixels

n_experiences = 10

# experience_array = [None] * n_experiences

immortal_count = 0.0

lifespans = []

diff_pixels = pixels - old_pixels
compressed = compress(diff_pixels)
# compressed_pa_np1 = temporal(compressed_pa)
compressed_pa_np1 = compressed

torch.autograd.set_detect_anomaly(True)

while True:
    print('count', count)
    print('immortal count', immortal_count)
    if immortal_count % 1000 == 0:
        torch.save(compress.state_dict(), 'compress.model')
        torch.save(temporal.state_dict(), 'temporal.model')
        torch.save(quality.state_dict(), 'quality.model')
    if game.gameState == "attract_mode" or game.gameState == "exploding":
        lifespans.append(count)
        print(lifespans)
        game.playGame([start])
        count = 0.0
    # start_time = time()
    pixels = pygame.surfarray.pixels3d(game.stage.screen)
    pixels = pixels[:,:,0]
    pixels = torch.from_numpy(pixels.astype(np.float32)).to(device)
    pixels = torch.reshape(pixels, (1, 1920, 1080))
    # print('preprocess time', time() - start_time)
    human_moves = pygame.event.get()




    # print(human_moves)
    # ai_move = np.random.choice(ai_moves)
    # ai_move_event = ai_moves_to_event[ai_move]
    # ai_move_h = np.zeros((1, len(ai_moves)))
    # ai_move_h[0][ai_move] = 1.0
    # ai_move_h_t = torch.Tensor(ai_move_h).to(device)
    # ai_move = torch.Tensor(ai_move)
    # ai_move_h = F.one_hot(ai_move, len(ai_moves))
    # start_time = time()
    # print(pixels.shape)
    diff_pixels = pixels - old_pixels
    compressed = compress(diff_pixels)


    sim_error = F.mse_loss(compressed.detach(), compressed_pa_np1)
    print('sim_error', sim_error)
    # sim_error.backward(retain_graph=True)
    # opt_temporal.step()
    # opt_temporal.zero_grad()

    # print(compressed)
    ai_move, q = generate(compressed)
    old_pixels = pixels
    print('q score', q)
    ai_move_event = ai_moves_to_event[ai_move]


    ai_move_h = np.zeros((1, len(ai_moves)))
    ai_move_h[0][ai_move] = 1.0
    ai_move_h_t = torch.Tensor(ai_move_h).to(device)
    compressed_pa = torch.concatenate([compressed, ai_move_h_t], dim=1)
    with torch.no_grad():
        compressed_pa_np1 = temporal(compressed_pa)





    # compressed_pa = torch.concatenate([compressed, ai_move_h_t], dim=1)
    # compressed_pa_np1 = temporal(compressed_pa)
    # compressed_q = quality(compressed_pa_np1)
    # print(compressed_q.shape)
    # print(compressed_q)
    # print('nn time', time() - start_time)
    # print(game.stage.spriteList)
    # ai_move = up
    # ai_move = generate(copy.deepcopy(game.stage.spriteList))

    game.playGame([ai_move_event]+human_moves)
    # game.playGame([ai_move])
    
    # actual_score = torch.Tensor([game.score]).to(device)
    actual_score = torch.Tensor([count]).to(device)
    print('actual_score', actual_score)
    error = F.mse_loss(q, actual_score)
    print('error', error)
    # error.backward(retain_graph=True)
    # opt.step()
    # opt.zero_grad()
    total_error = error + sim_error
    # total_error = error
    print('total_error', total_error)
    total_error.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()


    # print('average lifespan', np.mean(filter(lambda a: a != 0 and a != 1, lifespans)))
    count += 1.0
    immortal_count += 1.0

####
