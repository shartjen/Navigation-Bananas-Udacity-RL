import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# some parameters are now being import via init, better check which one is the actual value
BUFFER_SIZE = int(10e5) # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
# ALPHA = 0             # Exponent for deltaQ in memory
EMIN = 0.01             # bias for deltaQ to ensure small values still being visited
BETA = 0.6              # Annealing exponent for IS weights
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
DOPER = True            # False Experience Replay : True Prioritized ER
PBAR_MULT = 7           # Multiplier to pbar in PER acceptance
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentPER():
    """Interacts with and learns from the environment."""

    def __init__(self,alpha,beta,gamma,tau,LR,batch_size, state_size, action_size, seed, buffer_size = BUFFER_SIZE):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        
        # set parameter for ML
        self.set_parameters(gamma, tau, LR, batch_size)
        # Q-Network
        self.create_qnetworks(seed)
        # Replay memory
        self.create_ReplayBuffer(buffer_size, alpha, beta, EMIN, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def set_parameters(self, gamma, tau, LR, batch_size):
        # Base agent parameters
        self.gamma = gamma
        self.tau = tau
        self.LR = LR 
        self.batch_size = batch_size
        # Some debug flags
        self.DebugSample = False
        
    def create_qnetworks(self, seed):
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        
    def create_ReplayBuffer(self, buffer_size, alpha, beta, emin, seed):
        self.alpha = alpha
        self.beta = beta
        self.emin = EMIN
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(self.action_size, buffer_size, self.batch_size, alpha, emin, seed)
    
    def getdeltaQ(self, state, action, reward, next_state, done):
        """ Get difference between Q_target and Q_expected as in target and local """
        """ state, action ..  as [numpy.Variable] or torch): (s, a, r, s', done) """

        # print('GetdeltaQ : type(state) : ',type(state))

        # Cast to tensor if inputted as numpy
        if type(state) is np.ndarray:
            # print('Is numpy, casting types')
            state = torch.from_numpy(state).float().unsqueeze(0).to(device) 
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            action = torch.from_numpy(np.asarray(action)).long().unsqueeze(0).unsqueeze(0).to(device)
        
        # Get max predicted Q values (for next states) from target model
        Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_target = reward + (self.gamma * Q_target_next * (1 - done))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state).gather(1, action)
        # get the deltaQ to feed to memory
        deltaQ = Q_target - Q_expected
        # from Tensor to float, adding happens via float etc.
        deltaQ = deltaQ.detach().numpy()
        
        # Add emin as sign
        deltaQ += np.sign(deltaQ)*self.emin
        if np.any(deltaQ == 0):
            # replace zeroes bei self.emin (sign(-abs(deltaQ))+1) = 0 for all others
            deltaQ += (np.sign(-abs(deltaQ))+1)*self.emin
        
        return deltaQ
    
    def step(self, state, action, reward, next_state, done):
        """Proceeds all variable for doing an agents step"""
        """ state, action ..  as [numpy.Variable]): (s, a, r, s', done) """
        
        # Get Q-changes(deltaQ) to store in replay memory to prioritize through those
        deltaQ = self.getdeltaQ(state, action, reward, next_state, done)

        self.check_Q_larger_emin(deltaQ, 'step')
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, deltaQ, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                if DOPER:
                    experiences,exp_indices,probs = self.memory.sample()
                    states, actions, rewards, next_states, deltaQs, dones = experiences
                    self.DebugSampleStep(deltaQs, exp_indices, probs)
                else:
                    experiences,exp_indices,probs = self.memory.ERsample()
                    states, actions, rewards, next_states, deltaQs, dones = experiences                
                    self.DebugSampleStep(deltaQs, exp_indices, probs)

                self.learn(experiences,exp_indices, probs, self.gamma)
                
        self.EndOfEpisode()
        
    def DebugSampleStep(self, deltaQs, exp_indices, probs):
        
        if self.DebugSample:
            deltaQnp = np.transpose(deltaQs.detach().numpy())
            print('Step: Sampled Indices and associated deltaQs:  (mean(deltaQ) = {:7.5f}|| mean(abs(deltaQ)) = {:7.5f}) '.format(np.mean(deltaQnp),np.mean(np.abs(deltaQnp))))
            print(exp_indices)
            print(deltaQnp)
            print('step probs : ')
            print('MinabsP : {:11.9f} MaxabsP : {:11.9f} and meanP = {:11.9f}'.format(min(abs(probs)),max(abs(probs)),np.mean(probs)))
            print(probs)
            # ERdeltaQnp = np.transpose(ERdeltaQs.detach().numpy())
            # print('Learn: Sampled Indices and associated deltaQs as in ER:  (mean(deltaQ) = {:7.5f}|| mean(abs(deltaQ)) = {:7.5f}) '.format(np.mean(ERdeltaQnp),np.mean(np.abs(ERdeltaQnp))))
            # print(ERexp_indices)
            # print(ERdeltaQnp)

    def EndOfEpisode(self):
        """" Take care of any end of Epsiode tasks 
             for now we will just update SumQ in memory which has some minor inaccuracies over time """
             
        # self.memory.ReCalcSumQ()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #if DEBUG:print('State before torch: ',state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #if DEBUG:print('State after torch: ',state)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))
        
    def update_PER(self,experiences,exp_indices):
        """ Update experience in Replay Memory with new deltaQ """
        
        states, actions, rewards, next_states, deltaQs, dones = experiences
        NewQ = self.getdeltaQ(states, actions, rewards, next_states, dones)
        
        self.check_Q_larger_emin(NewQ, 'update_PER')
        self.memory.updatedeltaQ(NewQ,exp_indices)
        
        return
    
    def check_Q_larger_emin(self, Q, funcname):

        if np.any(abs(Q) < self.emin):
            print('-----------------------------------------')
            print('{} : abs(NewQ) < emin, this should not happen'.format(funcname))
            print('-----------------------------------------')
            print(Q)

    def learn(self, experiences, exp_indices, probs, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        printQs = False
        states, actions, rewards, next_states, deltaQs, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        if printQs:
            Q_exp = np.transpose(Q_expected.detach().numpy())
            print('Learn : Q Expected from current experiences: (mean = {:7.5f})'.format(np.mean(Q_exp)))
            print(Q_exp)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        ISweights = (probs/len(self.memory)) ** self.beta
        for param, weight in zip(self.qnetwork_local.parameters(), ISweights):
            param.grad.data *= weight
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

        # update priorities in PER for experiences processed
        self.update_PER(experiences, exp_indices)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

    def mem_print_summary(self):
        
        AllQ, AllP = self.memory.mem_print_summary()
        
        return AllQ, AllP



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha,  emin, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "deltaQ", "done"])
        self.alpha = alpha
        self.emin = emin
        self.seed = random.seed(seed)
        self.Sum_deltaQ = 0

    
    def mem_print_summary(self):
        
        print('Replay Buffer Summary :')
        print('Exponent for weights : ',self.alpha)
        print('Number of Elements in Buffer : ',len(self))
        
        AllQ = np.zeros(len(self))
        AllP = np.zeros(len(self))
        for e,i in zip(self.memory,np.arange(len(self))):
            AllQ[i] = e.deltaQ
        WeightedSumQ = sum(abs(AllQ)**self.alpha)            
        for e,i in zip(self.memory,np.arange(len(self))):
            AllP[i] = self.p_from_q(e.deltaQ)

        # print(AllQ)
        print('MinabsQ : {:11.9f} MaxabsQ : {:11.9f} and meanQ = {:11.9f} while mean(abs(Q)) = {:11.9f}'.format(min(abs(AllQ)),max(abs(AllQ)),np.mean(AllQ),np.mean(abs(AllQ))))
        print('With weighted sum from AllQ being : {} and stored one is : {}'.format(WeightedSumQ, self.Sum_deltaQ.item()))
        # print(AllQ)
        # print(abs(AllQ)**self.alpha)

        print('MinabsP : {:11.9f} MaxabsP : {:11.9f} and meanP = {:11.9f}'.format(min(abs(AllP)),max(abs(AllP)),np.mean(AllP)))
        print('With sum from AllP being : {} hopefully being 1.'.format(sum(AllP)))
        print('Max/pbar = {:6.4f}'.format(max(AllP)/np.mean(AllP)))

        return AllQ, AllP
    
    def add(self, state, action, reward, next_state, deltaQ, done):
        """Add a new experience to memory."""
        
        np.set_printoptions(precision=4)
        # print('Adding Element to buffer')# state : ',state)
        # print('Reward : ',reward,' DQ : ',deltaQ)
        if len(self) >= self.buffer_size:
            # print('Buffer Full - wait a second: Removing one element')
            # if full pop from left and correct sum
            epop = self.memory.popleft()
            self.Sum_deltaQ -= np.abs(epop.deltaQ.item()) ** self.alpha
        # append right and adjust sum
        e = self.experience(state, action, reward, next_state, deltaQ, done)
        self.memory.append(e)
        self.Sum_deltaQ += np.abs(deltaQ.item()) ** self.alpha
        
        # self.mem_print_summary()
        
        # wdeltaQ = np.abs(deltaQ)**self.alpha
        # print('All updates done, Sum of weighted Qs : {:6.4f} as : {:6.4f} was added.'.format(self.Sum_deltaQ,wdeltaQ)) 
        # print('Replay Memory contains : ',len(self.memory),' elements')
        # print('Show Memory : ',self.memory)
        # self.printdeltaQs()
        
    def printdeltaQs(self):
        ShowQandP = False
        np.set_printoptions(precision=4)
        if ShowQandP:print('Showing Qs')
        probs = np.zeros(len(self))
        Qs = np.zeros(len(self))

        for e,i in zip(self.memory,np.arange(len(self))):
            Qs[i] = e.deltaQ
        
        if ShowQandP:print(Qs)
        if ShowQandP:print('Associated Probs : ')
        probs = np.zeros(len(self))
        for e,i in zip(self.memory,np.arange(len(self))):
            prob = self.p_from_q(e.deltaQ)
            probs[i] = prob
            
        if ShowQandP:print(probs)
        print('MinP : {:8.6f} MaxP : {:8.6f} and Pbar = {:8.6f} should be 1/n = {:8.6f} with n = {} '.format(min(probs),max(probs),np.mean(probs),1/len(self),len(self)))
        print('Max/pbar = {:6.4f}'.format(max(probs)/np.mean(probs)))
        print('MinQ : {:8.6f} MaxQ : {:8.6f} and Qbar = {:8.6f}'.format(min(Qs),max(Qs),np.mean(Qs)))
        print('MinabsQ : {:8.6f} MaxabsQ : {:8.6f} and absQbar = {:8.6f}'.format(min(abs(Qs)),max(abs(Qs)),np.mean(abs(Qs))))
        
        return
    
    def p_from_q(self, q):
        return np.abs(q)**self.alpha/self.Sum_deltaQ
    
    def ReCalcSumQ(self):
        self.Sum_deltaQ = sum([e.deltaQ for e in self.memory]).item()
    
    def updatedeltaQ(self,NewQ,exp_indices):
        
        debugupdate = False
        if debugupdate:
            print('PER updates following indicies and New Qs:')
            print(np.transpose(NewQ))
            print(exp_indices)
        
        for ei,q in zip(exp_indices,NewQ):
            if debugupdate:print('Updating index : {} with new Q : {:6.4f} while previous value is {:6.4f}'.format(ei, q.item(),self.memory[ei].deltaQ.item()))
            self.Sum_deltaQ += (abs(q)**self.alpha-abs(self.memory[ei].deltaQ)**self.alpha)
            self.memory[ei] = self.memory[ei]._replace(deltaQ = q)
            
        
        return
    
    def drawsample(self, numdraws):
        """ Draw prioritized sample according to probabilities by deltaQs stored 
        
        Params
        ======
            numdraws (int): number of experiences to sample """

        Sampling_Debug_DS = False
        experiences = []
        drawn_indices = []
        probs = []
        # Keeping track of number accepted and rejected
        numacc = 0
        numre = 0
        n = len(self)
        # Sampling with acceptance until numdraws draws are accepted
        for i in np.arange(numdraws):
            accepted = False
            while not accepted:
                curdraw = np.random.randint(0,n)
                
                # print('Curdraw : ',curdraw,' while ',drawn_indices,' have been drawn.')
                if curdraw not in drawn_indices:
                    e = self.memory[curdraw]
                    p = self.p_from_q(e.deltaQ).item()
                    # accept with prob=1 if p > 2*pbar or with prob = p/(2*pbar), PBAR_MULT is being 2 in this case
                    accepted = (p/PBAR_MULT*n > np.random.uniform())
                    if accepted:
                        numacc += 1
                        drawn_indices.append(curdraw)
                        experiences.append(e)
                        probs.append(p)
                        # print('Accepted {} on {}'.format(curdraw,i))
                    else:
                        numre += 1
                        # print('Rejected {} on {}'.format(curdraw,i))
                        
        if Sampling_Debug_DS:
            print('Final Sample : (acc : {} -- rej : {} )'.format(numacc,numre))
            # print(drawn_indices)
            np.set_printoptions(precision=4)
            # print(np.array(probs).transpose())
            
        return experiences,drawn_indices,np.asarray(probs)
        
    def ERsample(self):
        """Randomly sample a batch of experiences from memory."""
        exp_indices = random.sample(range(len(self.memory)), k=self.batch_size)
        experiences = [self.memory[i] for i in exp_indices]
        probs = np.ones(self.batch_size)/len(self)
        # experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        deltaQs = torch.from_numpy(np.vstack([e.deltaQ for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, deltaQs, dones), exp_indices, probs

    def sample(self):
        """Prioritized sample a batch of experiences from memory."""
        
        Sampling_Debug_S = False
        if Sampling_Debug_S:
            print('------------------------------------------------')
            print('Sampling Right now')
            self.printdeltaQs()

        experiences,exp_indices,probs = self.drawsample(numdraws=self.batch_size)
        # experiences = random.sample(self.memory, k=self.batch_size)
        if Sampling_Debug_S:print('len experiences = ',len(experiences))#,' and type : ',type(experiences))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        deltaQs = torch.from_numpy(np.vstack([e.deltaQ for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, deltaQs, dones), exp_indices, probs

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)