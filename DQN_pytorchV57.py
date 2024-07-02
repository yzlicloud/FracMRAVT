import os
import numpy as np
import glob
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from collections import OrderedDict
from frac2 import AdaFm
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import copy

DISCRETE_ACTIONS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0]
])

Transition = namedtuple('Transition', ['state', 'action_idx', 'reward', 'next_state'])


class ConvNet(nn.Module):
    """docstring for Net"""

    def __init__(self, observe_type='Color'):
        super(ConvNet, self).__init__()
        if observe_type == 'Color':
            self.conv1 = nn.Conv2d(3, 32, 3, 2,
                                   padding=1, bias=False)
        elif observe_type == 'Depth':
            self.conv1 = nn.Conv2d(1, 32, 3, 2,
                                   padding=1, bias=False)
        elif observe_type == 'RGBD':
            self.conv1 = nn.Conv2d(4, 32, 3, 2,
                                   padding=1, bias=False)
        else:
            raise RuntimeError('=> Unsupported observation type!')

        self.bn1 = nn.BatchNorm2d(32, eps=1e-6, momentum=0.05)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-6, momentum=0.05)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)
        self.conv4 = nn.Conv2d(128, 128, 1, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)

        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, len(DISCRETE_ACTIONS))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output += self.shortcut(x)
        output = F.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, observe_type='Color', block=None, num_blocks=None):
        super(ResNet, self).__init__()
        self.in_planes = 32
        if observe_type == 'Color':
            self.conv1 = nn.Conv2d(3, 32, 3, 2,
                                   padding=1, bias=False)
        elif observe_type == 'Depth':
            self.conv1 = nn.Conv2d(1, 32, 3, 2,
                                   padding=1, bias=False)
        elif observe_type == 'RGBD':
            self.conv1 = nn.Conv2d(4, 32, 3, 2,
                                   padding=1, bias=False)
        else:
            raise RuntimeError('=> Unsupported observation type!')
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)
        self.fc1 = nn.Linear(256 * block.expansion * 4 * 4, 2048)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, len(DISCRETE_ACTIONS))

    def _make_layer(self, block=None, planes=None, num_blocks=None, stride=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class ConvLSTM(nn.Module):
    def __init__(self, observe_type='Color'):
        super(ConvLSTM, self).__init__()
        if observe_type == 'Color':
            self.conv1 = nn.Conv2d(3, 32, 3, 2)
        elif observe_type == 'Depth':
            self.conv1 = nn.Conv2d(1, 32, 3, 2)
        elif observe_type == 'RGBD':
            self.conv1 = nn.Conv2d(4, 32, 3, 2)
        else:
            raise RuntimeError('=> Unsupported observation type!')

        self.bn1 = nn.BatchNorm2d(32, eps=1e-6, momentum=0.05)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-6, momentum=0.05)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)
        self.conv4 = nn.Conv2d(128, 128, 1, 1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-6, momentum=0.05)

        self.fc1 = nn.Linear(128 * 5 * 5, 512)

        self.lstm_input_size = 512
        self.lstm_hidden_size = 1024
        self.lstm_layers = 2
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, num_layers=self.lstm_layers)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, len(DISCRETE_ACTIONS))

    def forward(self, x, state_tuple_in):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if state_tuple_in is None:
            # batch_size = x.shape[0]
            state_tuple_in = (torch.randn(self.lstm_layers, 1, self.lstm_hidden_size).cuda(),
                              torch.randn(self.lstm_layers, 1, self.lstm_hidden_size).cuda())
        else:
            state_tuple_in = (torch.tensor(state_tuple_in[0]).cuda(),
                              torch.tensor(state_tuple_in[1]).cuda())

        x = torch.unsqueeze(x, dim=1)
        x, state_tuple_out = self.lstm(x, state_tuple_in)
        x = torch.squeeze(x, dim=1)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output, state_tuple_out

    def reset(self):
        return


class DQN(object):
    """DQN algorithm"""

    def __init__(self, name='dqn', log_dir='log', observe_type='Color', backbone='ConvNet',
                 action_type='Discrete', actuator_type='Position',
                 replay_buffer_size=50000, batch_size=2, lr=1e-5,
                 gamma=0.99, epsilon_factor=0.0, update_interval=1000,
                 restore=True, is_training=True):
        super(DQN, self).__init__()
        self.name = name
        assert action_type in ['Discrete', 'Continuous']
        self.action_type = action_type
        assert actuator_type in ['Position', 'Velocity', 'Force']
        self.actuator_type = actuator_type
        assert backbone in ['ConvNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ConvLSTM']
        self.backbone = backbone

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_factor = epsilon_factor
        self.update_interval = update_interval
        self.replay_buffer_size = replay_buffer_size
        self.observe_type = observe_type
        self.logdir = log_dir

        self.episode_counter = 0
        self.learn_step_counter = 0
        self.memory_total_counter = 0
        self.memory_counter = 0
        self.memory_counter1 = 0
        self.memory_counter2 = 0
        self.memory_counter3 = 0
        self.memory_counter4 = 0
        self.memory_counter5 = 0
        self.memory_counter6 = 0
        self.memory_counter7 = 0
        self.memory_counter8 = 0
        self.memory_counter9 = 0
        self.memory_counter10 = 0
        self.memory_counter11 = 0


        if self.backbone == 'ConvNet':
            self.eval_net = ConvNet(observe_type=observe_type).cuda()
            self.target_net = ConvNet(observe_type=observe_type).cuda()
            self.net_final = ConvNet(observe_type=observe_type).cuda()
            self.net_final1 = ConvNet(observe_type=observe_type).cuda()
        elif self.backbone == 'ConvLSTM':
            self.eval_net = ConvLSTM(observe_type=observe_type).cuda()
            self.target_net = ConvLSTM(observe_type=observe_type).cuda()
            self.net_final = ConvNet(observe_type=observe_type).cuda()
            self.net_final1 = ConvNet(observe_type=observe_type).cuda()
        elif self.backbone == 'ResNet18':
            self.eval_net = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[2, 2, 2, 2]).cuda()
            self.target_net = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[2, 2, 2, 2]).cuda()
            self.net_final = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[2, 2, 2, 2]).cuda()
            self.net_final1 = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[2, 2, 2, 2]).cuda()
        elif self.backbone == 'ResNet34':
            self.eval_net = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[3, 4, 6, 3]).cuda()
            self.target_net = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[3, 4, 6, 3]).cuda()
            self.net_final = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[3, 4, 6, 3]).cuda()
            self.net_final1 = ResNet(observe_type=observe_type, block=BasicBlock, num_blocks=[3, 4, 6, 3]).cuda()
        elif self.backbone == 'ResNet50':
            self.eval_net = ResNet(observe_type=observe_type, block=Bottleneck, num_blocks=[3, 4, 6, 3]).cuda()
            self.target_net = ResNet(observe_type=observe_type, block=Bottleneck, num_blocks=[3, 4, 6, 3]).cuda()
            self.net_final = ResNet(observe_type=observe_type, block=Bottleneck, num_blocks=[3, 4, 6, 3]).cuda()
            self.net_final1 = ResNet(observe_type=observe_type, block=Bottleneck, num_blocks=[3, 4, 6, 3]).cuda()

        else:
            raise ValueError('Unsupported backbone type!')

        if restore:
            model_path = self.get_latest_model_path()
            if os.path.exists(model_path):
                print('=> restoring model from {}'.format(model_path))
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
                # checkpoint = torch.load(model_path)
                self.eval_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                self.net_final.load_state_dict(checkpoint['model_state_dict'])
                self.net_final1.load_state_dict(checkpoint['model_state_dict'])

                self.episode_counter = checkpoint['episodes']
                self.learn_step_counter = checkpoint['steps'] + 1
            else:
                raise ValueError('No model file is found in {}'.format(model_path))

        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.memory = []
        self.memory1 = []
        self.memory2 = []
        self.memory3 = []
        self.memory4 = []
        self.memory5 = []
        self.memory6 = []
        self.memory7 = []
        self.memory8 = []
        self.memory9 = []
        self.memory10 = []
        self.memory11 = []



        self.meta_batch_dataset = []


        # self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        # self.optimizer_inner = optim.Adam(self.net_final.parameters(), lr=self.lr)
        self.optimizer = AdaFm(self.eval_net.parameters(), lr=self.lr)

        self.optimizer_inner = AdaFm(self.net_final.parameters(), lr=self.lr)
        
        self.optimizer_inner_1 = optim.Adam(self.net_final1.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.pre_hidden_state_tuple = None
        self.pre_batch_hidden_state_tuple = None

        self.is_training = is_training
        if not self.is_training:
            self.eval_net.eval()
            self.target_net.eval()
        else:
            self.summary_writer = SummaryWriter(log_dir=self.logdir)

    def choose_action(self, state, env):
        state = torch.from_numpy(np.transpose(state, (2, 0, 1)))
        state = torch.unsqueeze(torch.Tensor(state), 0)

        if np.random.rand() >= self.epsilon_factor:
            if self.backbone == 'ConvLSTM':
                # greedy policy
                action_value, hidden_state_tuple = self.eval_net.forward(state.cuda(), self.pre_hidden_state_tuple)
                self.pre_hidden_state_tuple = [hidden_state_tuple[0].detach(),
                                               hidden_state_tuple[1].detach()]
                indx = torch.argmax(action_value, 1)[0].item()
            else:
                # greedy policy
                action_value = self.eval_net.forward(state.cuda())
                indx = torch.argmax(action_value, 1)[0].item()
        else:
            # random policy
            indx = np.random.randint(0, len(DISCRETE_ACTIONS))

        action = DISCRETE_ACTIONS[indx]

        return indx, action

    def store_transition(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory.append(transition)
        self.memory_counter += 1
        self.memory_total_counter+= 1


    def store_transition1(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory1.append(transition)
        self.memory_counter1 += 1
        self.memory_total_counter+= 1

    def store_transition2(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory2.append(transition)
        self.memory_counter2 += 1
        self.memory_total_counter+= 1

    def store_transition3(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory3.append(transition)
        self.memory_counter3 += 1
        self.memory_total_counter+= 1

    def store_transition4(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory4.append(transition)
        self.memory_counter4 += 1
        self.memory_total_counter+= 1

    def store_transition5(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory5.append(transition)
        self.memory_counter5 += 1
        self.memory_total_counter+= 1


    def store_transition6(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory6.append(transition)
        self.memory_counter6 += 1
        self.memory_total_counter+= 1

    def store_transition7(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory7.append(transition)
        self.memory_counter7 += 1
        self.memory_total_counter+= 1

    def store_transition8(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory8.append(transition)
        self.memory_counter8 += 1
        self.memory_total_counter+= 1


    def store_transition9(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory9.append(transition)
        self.memory_counter9 += 1
        self.memory_total_counter+= 1


    def store_transition10(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory10.append(transition)
        self.memory_counter10 += 1
        self.memory_total_counter+= 1


    def store_transition11(self, state, action_idx, reward, next_state):
        transition = Transition(state, action_idx, reward, next_state)
        self.memory11.append(transition)
        self.memory_counter11 += 1
        self.memory_total_counter+= 1


    def forward(loss):
        self.optimizer_inner.zero_grad()
        # loss.backward(retain_graph=True)
        # optimizer_inner.step()
        # loss = self.net_pi(query_x, query_y)
        grads_pi = torch.autograd.grad(loss, self.net_final.parameters(), retain_graph=True)
        loss.backward(retain_graph=True)
        self.optimizer_inner.step()
        return grads_pi


    def write_grads_inner(self, dummy_loss, sum_grads_pi):
        hooks = []
        for i, v in enumerate(net_final.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]
            hooks.append(v.register_hook(closure()))
        optimizer_inner.zero_grad()
        dummy_loss.backward(retain_graph=True)
        optimizer_inner.step()
        for h in hooks:
            h.remove()


    def write_grads_inner_1(self, dummy_loss, sum_grads_pi):
        hooks = []
        for i, v in enumerate(net_final1.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]
            hooks.append(v.register_hook(closure()))
        optimizer_inner_1.zero_grad()
        dummy_loss.backward(retain_graph=True)
        optimizer_inner_1.step()
        for h in hooks:
            h.remove()




    def write_grads_outer(self, dummy_loss, sum_grads_pi):
        hooks = []
        for i, v in enumerate(self.eval_net.parameters()):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]
            hooks.append(v.register_hook(closure()))
        self.optimizer.zero_grad()
        # dummy_loss.backward(retain_graph=True)
        dummy_loss.backward()
        # dummy_loss.backward(retain_graph=True)
        self.optimizer.step()

            # if you do NOT remove the hook, the GPU memory will explode!!!
        for h in hooks:
            h.remove()



    def learn(self):
        # update the parameters
        self.learn_step_counter += 1

        Train_task=[self.memory, self.memory1, self.memory2, self.memory3, self.memory4, self.memory5, self.memory6, self.memory7, self.memory8, self.memory9, self.memory10, self.memory11]
        Train_task_club=random.sample(Train_task, 5)


        Eval_task=[self.memory, self.memory1, self.memory2, self.memory3, self.memory4, self.memory5, self.memory6, self.memory7, self.memory8, self.memory9, self.memory10, self.memory11]
        Eval_task_club=random.sample(Eval_task, 5)


        samples_train= random.sample(Train_task_club[0], self.batch_size)
        for i in range(1, len(Train_task_club)):
            samples_train.extend(random.sample(Train_task_club[i], self.batch_size))


        samples_eval= random.sample(Eval_task_club[0], self.batch_size)
        for i in range(1, len(Eval_task_club)):
            samples_eval.extend(random.sample(Eval_task_club[i], self.batch_size))


        # samples_train = random.sample(self.memory, self.batch_size)
        # samples_eval = random.sample(self.memory, self.batch_size)


        def forward(loss):
            optimizer_inner.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer_inner.step()
            # loss = self.net_pi(query_x, query_y)
            grads_pi = torch.autograd.grad(loss, net_final.parameters(), retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer_inner.step()
            return grads_pi
            
        def write_grads_inner(dummy_loss, sum_grads_pi):
            hooks = []
            for i, v in enumerate(net_final.parameters()):
                def closure():
                    ii = i
                    return lambda grad: sum_grads_pi[ii]
                hooks.append(v.register_hook(closure()))
            self.optimizer_inner.zero_grad()
            dummy_loss.backward(retain_graph=True)
            self.optimizer_inner.step()
            for h in hooks:
                h.remove()




        def write_grads_inner_1(dummy_loss, sum_grads_pi):
            hooks = []
            for i, v in enumerate(net_final1.parameters()):
                def closure():
                    ii = i
                    return lambda grad: sum_grads_pi[ii]
                hooks.append(v.register_hook(closure()))
            self.optimizer_inner_1.zero_grad()
            dummy_loss.backward(retain_graph=True)
            self.optimizer_inner_1.step()
            for h in hooks:
                h.remove()




        def write_grads( dummy_loss, sum_grads_pi):
            hooks = []
            for i, v in enumerate(self.eval_net.parameters()):
                def closure():
                    ii = i
                    return lambda grad: sum_grads_pi[ii]
                hooks.append(v.register_hook(closure()))
            self.optimizer.zero_grad()
            # dummy_loss.backward(retain_graph=True)
            dummy_loss.backward()
            # dummy_loss.backward(retain_graph=True)
            self.optimizer.step()

                # if you do NOT remove the hook, the GPU memory will explode!!!
            for h in hooks:
                h.remove()



        # batch_states, batch_action_idxs, batch_rewards, batch_next_states = map(np.array, zip(*meta_batch_dataset))
        batch_states, batch_action_idxs, batch_rewards, batch_next_states = map(np.array, zip(*samples_train))

        batch_states = torch.FloatTensor(np.transpose(batch_states, (0, 3, 1, 2))).cuda()
        batch_action_idxs = torch.LongTensor(batch_action_idxs).unsqueeze(1).cuda()
        batch_rewards = torch.FloatTensor(batch_rewards).cuda()
        batch_next_states = torch.FloatTensor(np.transpose(batch_next_states, (0, 3, 1, 2))).cuda()




        eval_batch_states, eval_batch_action_idxs, eval_batch_rewards, eval_batch_next_states = map(np.array, zip(*samples_eval))

        eval_batch_states = torch.FloatTensor(np.transpose(eval_batch_states, (0, 3, 1, 2))).cuda()
        eval_batch_action_idxs = torch.LongTensor(eval_batch_action_idxs).unsqueeze(1).cuda()
        eval_batch_rewards = torch.FloatTensor(eval_batch_rewards).cuda()
        eval_batch_next_states = torch.FloatTensor(np.transpose(eval_batch_next_states, (0, 3, 1, 2))).cuda()



        if self.backbone == 'ConvLSTM':
            action_value, _ = self.net_final(batch_states, self.pre_hidden_state_tuple)
            q_eval = action_value.gather(1, batch_action_idxs).squeeze(1)
            target_value, _ = self.target_net(batch_next_states, self.pre_hidden_state_tuple)
            q_next = torch.max(target_value.detach(), 1)[0]
        else:
            q_eval = self.net_final(batch_states).gather(1, batch_action_idxs).squeeze(1)
            q_next = torch.max(self.target_net(batch_next_states).detach(), 1)[0]
        q_target = batch_rewards + self.gamma * q_next

        loss = self.loss_func(q_eval, q_target)

        
        ###  1
        # grads = torch.autograd.grad(loss, self.net_final.parameters(), retain_graph=True)
        # write_grads_inner(loss, grads)
        ###  2
        self.optimizer_inner.zero_grad()
        loss.backward()
        self.optimizer_inner.step()





        if self.backbone == 'ConvLSTM':
            action_value, _ = self.net_final(eval_batch_states, self.pre_hidden_state_tuple)
            q_eval = action_value.gather(1, eval_batch_action_idxs).squeeze(1)
            target_value, _ = self.target_net(eval_batch_next_states, self.pre_hidden_state_tuple)
            q_next = torch.max(target_value.detach(), 1)[0]
        else:
            q_eval = self.net_final(eval_batch_states).gather(1, eval_batch_action_idxs).squeeze(1)
            q_next = torch.max(self.target_net(eval_batch_next_states).detach(), 1)[0]
        q_target = eval_batch_rewards + self.gamma * q_next
        # loss = self.loss_func(q_eval, q_target.cuda())
        loss1 = self.loss_func(q_eval, q_target)
        ###  1
        # grads = torch.autograd.grad(loss1, net_final.parameters(), retain_graph=True)
        # write_grads_inner(loss1, grads)
        ##  2
        self.optimizer_inner.zero_grad()
        loss1.backward()
        self.optimizer_inner.step()










        if self.backbone == 'ConvLSTM':
            action_value, _ = self.net_final(eval_batch_states, self.pre_hidden_state_tuple)
            q_eval = action_value.gather(1, eval_batch_action_idxs).squeeze(1)
            target_value, _ = self.target_net(eval_batch_next_states, self.pre_hidden_state_tuple)
            q_next = torch.max(target_value.detach(), 1)[0]
        else:
            q_eval = self.net_final(eval_batch_states).gather(1, eval_batch_action_idxs).squeeze(1)
            q_next = torch.max(self.target_net(eval_batch_next_states).detach(), 1)[0]
        q_target = eval_batch_rewards + self.gamma * q_next
        # loss = self.loss_func(q_eval, q_target.cuda())
        loss2 = self.loss_func(q_eval, q_target)
        ###  1
        # grads = torch.autograd.grad(loss1, net_final.parameters(), retain_graph=True)
        # write_grads_inner(loss1, grads)
        ##  2
        grads = torch.autograd.grad(loss2, self.net_final.parameters(), retain_graph=True)
        ###  1
        # write_grads(loss, grads)
        self.write_grads_outer(loss2, grads)









        for i in range(5):
            if self.backbone == 'ConvLSTM':
                action_value, _ = self.eval_net(eval_batch_states, self.pre_hidden_state_tuple)
                q_eval = action_value.gather(1, eval_batch_action_idxs).squeeze(1)
                target_value, _ = self.target_net(eval_batch_next_states, self.pre_hidden_state_tuple)
                q_next = torch.max(target_value.detach(), 1)[0]
            else:
                q_eval = self.eval_net(eval_batch_states).gather(1, eval_batch_action_idxs).squeeze(1)
                q_next = torch.max(self.target_net(eval_batch_next_states).detach(), 1)[0]
            q_target = eval_batch_rewards + self.gamma * q_next
            # loss = self.loss_func(q_eval, q_target.cuda())
            loss3 = self.loss_func(q_eval, q_target)
            ###  1
            # grads = torch.autograd.grad(loss1, net_final.parameters(), retain_graph=True)
            # write_grads_inner(loss1, grads)
            ##  2
            self.optimizer.zero_grad()
            loss3.backward()
            self.optimizer.step()









        self.summary_writer.add_scalar('loss/value_loss', loss3, self.learn_step_counter)
        self.summary_writer.add_scalar('Q/action_value', q_eval.sum() / 5*self.batch_size, self.learn_step_counter)

    def get_latest_model_path(self):
        model_paths = glob.glob(os.path.join(self.logdir, 'QNet_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), self.logdir))
            created_times = [os.path.getmtime(path) for path in model_paths]
            latest_path = model_paths[np.argmax(created_times)]
            print('=> the latest model path: {}'.format(latest_path))
            return latest_path
        else:
            raise ValueError('No pre-trained model found!')

    def reset(self, is_record='', record_path=''):
        if self.backbone == 'ConvLSTM':
            self.eval_net.reset()
            self.target_net.reset()
        return

    def state_transform(self, state, env):
        """
        Normalize depth channel of state
        :param state: the state of observation
        :param env: the simulation env
        :return: norm_image
        """
        if self.observe_type == 'Color':
            norm_image = state
        elif self.observe_type == 'Depth':
            norm_image = state / env.cam_far_distance
        elif self.observe_type == 'RGBD':
            image = state[:, :, :3]
            depth = state[:, :, -1] / env.cam_far_distance
            norm_image = np.append(image, np.expand_dims(depth, 2), axis=2)
        else:
            raise ValueError('Unsupported observation type!')
        return norm_image

    @staticmethod
    def action_pos_2_force(pos_action, env):
        return 2 * env.chaser_mass * pos_action / (env.delta_time ** 2)

    @staticmethod
    def action_vel_2_force(vel_action, env):
        return vel_action / env.delta_time



