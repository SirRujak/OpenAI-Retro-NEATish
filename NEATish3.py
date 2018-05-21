 18 lines (16 sloc) 645 Bytes
# -*- coding: utf-8 -*-
__author__ = 'Chason'



## Node.py
class Node(object):
    """The node of evolutionary neural network.
    Attributes:
        id: The unique identification number of the node.
        value: The value of the node, ranging from 0 to 1.
        tag: The tag of the node, such as input node, hidden node, and output node.
    """
    def __init__(self, id, tag = '', value = 0):
        self.id = id
        self.value = value
        self.tag = tag

    def __eq__(self, other):
        """If the node id and tag are the same, then the two nodes are considered the same."""
return self.id == other.id and self.tag == other.tag


## NEAT.py
import sys
import math

class NEAT(object):
    """An evolutionary neural network called 'NeuroEvolution of Augmenting Topologies(NEAT)'
    Attributes:
        id: The unique identification number of NEAT.
        input_size: Input size of NEAT.
        output_size: Output size of NEAT.
        fitness: Adaptability of NEAT in the environment.
        node_count: The total number of nodes in NEAT.
        bias_node: Bias node in NEAT.
        input_nodes: Input nodes in NEAT.
        hidden_nodes: Hidden nodes in NEAT.
        output_nodes: Output nodes in NEAT.
        connections: Connections in NEAT.
        connection_list(static): A list of all different connections.
    """
    connection_list = []
    @staticmethod
    def sigmoid(z):
        """"Sigmoid activate function"""
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def tanh(z):
        """"Tanh activate function"""
        return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

    @staticmethod
    def probability(p):
        """Input range:0 <= p <= 1;The probability of returning True is p, and False is 1 - p"""
        return random.random() <= p

    def __init__(self, id, input_size, output_size, offspring=False):
        self.id = id
        self.input_size = input_size
        self.output_size = output_size
        self.fitness = 0
        self.node_count = 1
        self.bias_node = Node(id=0, tag='Bias Node', value=1)

        # input node
        self.input_nodes = []
        for i in range(input_size):
            self.input_nodes.append(Node(id=self.node_count, tag='Input Node'))
            self.node_count += 1

        # hidden node
        self.hidden_nodes = []

        # output node
        self.output_nodes = []
        for j in range(output_size):
            self.output_nodes.append(Node(id=self.node_count, tag='Output Node'))
            self.node_count += 1

        # connection
        self.connections = []
        if not offspring:
            for i in range(input_size):
                for j in range(output_size):
                    self.add_connection(self.input_nodes[i], self.output_nodes[j])
            for j in range(output_size):
                self.add_connection(self.bias_node, self.output_nodes[j])

    def __eq__(self, other):
        return self.id == other.id

    def connection_count(self):
        """Counts the number of connections enabled in NEAT."""
        count = 0
        for con in self.connections:
            if con.enable:
                count += 1
        return count
    def show_info(self):
        print "\tGenome %d(fitness = %.2f):Total Nodes:%d\tHidden Nodes:%d\tEnabled Connections:%d" % (
            self.id, self.fitness, self.node_count, len(self.hidden_nodes), self.connection_count())

    def show_structure(self, info_only=False):
        if info_only:
            print "Genome %d(fitness = %.2f):Total Nodes:%d\tHidden Nodes:%d\tEnabled Connections:%d" % (
                self.id, self.fitness, self.node_count, len(self.hidden_nodes), self.connection_count())
            return
        print "Genome %d(fitness = %.2f):"%(self.id, self.fitness)
        print "\tTotal Nodes:%d\tHidden Nodes:%d"%(self.node_count, len(self.hidden_nodes))
        print "\tEnabled Connections(%d):"%self.connection_count()
        for con in self.connections:
            print "\t\t[%s %d] = %.2f\t**[%6.2f]**\t[%s %d] = %.2f\tEnable = %s\tInnovation = %d"%(
                    con.input.tag, con.input.id, con.input.value,
                    con.weight,
                    con.output.tag, con.output.id, con.output.value,
                    con.enable, con.innovation)
        print

    def update_node(self, id):
        """Use sigmoid function to compute node values, ignoring the connections that are not enabled."""
        sum = 0
        for con in self.connections:
            if con.enable and con.output.id == id:
                sum += con.input.value * con.weight
        return self.tanh(sum)

    def forward_propagation(self):
        for hid in self.hidden_nodes:
            hid.value = self.update_node(hid.id)

        for out in self.output_nodes:
            out.value = self.update_node(out.id)

    def get_max_output_index(self):
        res = 0
        max_value = 0
        for inx, output in enumerate(self.output_nodes):
            if output.value > max_value:
                max_value = output.value
                res = inx
        return res

    def get_legal_output(self, board, col):
        res_r, res_c = 0, 0
        max_value = -sys.maxint
        for inx, output in enumerate(self.output_nodes):
            r, c = inx / col, inx % col
            if output.value > max_value and board[r][c] == 0:
                max_value = output.value
                res_r, res_c = r, c
        return res_r, res_c

    @staticmethod
    def get_innovation(connection):
        """Get innovation number and ensure that the same connection structure has the same innovation number."""
        # check existed connection
        for con in NEAT.connection_list:
            if con == connection:
                return con.innovation
        # new innovation number
        res = len(NEAT.connection_list)
        NEAT.connection_list.append(connection)
        return res

    def get_node_by_id(self, id):
        nodes = self.hidden_nodes
        if id == 0:
            return self.bias_node
        elif id <= len(self.input_nodes):
            nodes = self.input_nodes
        elif id <= self.input_size + self.output_size:
            nodes = self.output_nodes
        for node in nodes:
            if node.id == id:
                return node

    def add_connection_id(self, input_node_id, output_node_id, weight=None, enable=True):
        """Add a new connection by nodes id. If the weights are not set, the weights are set at random."""
        input_node = self.get_node_by_id(input_node_id)
        output_node = self.get_node_by_id(output_node_id)
        self.add_connection(input_node, output_node, weight, enable)

    def add_connection(self, input_node, output_node, weight=None, enable=True):
        """Add a new connection. If the weights are not set, the weights are set at random."""
        if weight == None:
            con = Connection(input=input_node, output=output_node, enable=enable)
            con.random_weight()
            con.innovation = NEAT.get_innovation(con)
        else:
            con = Connection(input=input_node, output=output_node, weight=weight)
            con.innovation = NEAT.get_innovation(con)
        # Insert sorting
        inx = len(self.connections) - 1
        while inx >= 0:
            if self.connections[inx].innovation <= con.innovation:
                break
            inx -= 1
        self.connections.insert(inx + 1, con)

    def add_hidden_node(self, tag="Hidden Node"):
        """Add a new hidden node. The default tag is 'Hidden Node'."""
        node = Node(self.node_count, tag=tag)
        self.hidden_nodes.append(node)
        self.node_count += 1
        return node

    def is_connection_exist(self, input, output):
        """Returns true if the connection already exists, otherwise returns false."""
        for con in self.connections[(self.input_size + 1 ) * self.output_size:]:
            if input.id == con.input.id and output.id == con.output.id:
                return True
        return False

    def mutation(self, new_node=True):
        """Let the neural network randomly mutate."""
        if self.probability(0.99):
            # modify connection weight
            for con in self.connections:
                if self.probability(0.9):
                    # uniformly perturb
                    con.weight += random.uniform(-3, 3)
                else:
                    # assign a new random weight
                    con.random_weight()

        if self.probability(0.05):
            # add a new connection
            for hid in self.hidden_nodes:
                # consider bias node
                if self.probability(0.5):
                    if not self.is_connection_exist(self.bias_node, hid):
                        self.add_connection(self.bias_node, hid)
                        break
                # search input nodes
                if self.probability(0.5):
                    for node in self.input_nodes:
                        if not self.is_connection_exist(node, hid):
                            self.add_connection(node, hid)
                            return
                # search hidden nodes
                if self.probability(0.5):
                    for hid2 in self.hidden_nodes:
                        if hid.id != hid2.id and not self.is_connection_exist(hid, hid2):
                            self.add_connection(hid, hid2)
                            return
                # search output nodes
                if self.probability(0.5):
                    for node in self.output_nodes:
                        if not self.is_connection_exist(hid, node):
                            self.add_connection(hid, node)
                            return

        if new_node and self.probability(0.02):
            # add a new node
            con = random.choice(self.connections)
            con.enable = False
            node = self.add_hidden_node()
            self.add_connection(con.input, node, 1)
            self.add_connection(node, con.output, con.weight)



## Environment.py
import random
import copy
import pickle
import os
import sys

class Environment(object):
    """This is an ecological environment that can control the propagation of evolutionary neural networks.
    Attributes:
        input_size: The input size of the genomes.
        output_size: The output size of the genomes.
        init_population: The initial population of genomes in the environment.
        max_generation: The maximum number of genomes generations
        genomes: The list of NEAT(NeuroEvolution of Augmenting Topologies)
    """
    def __init__(self,input_size, output_size, init_population, max_generation, comp_threshold, avg_comp_num,
                 mating_prob, copy_mutate_pro, self_mutate_pro,excess, disjoint, weight, survive, task,
                 file_name = None):
        self.input_size = input_size
        self.output_size = output_size
        self.population = init_population
        self.evaluation = init_population
        self.max_generation = max_generation
        self.next_generation = []
        self.outcomes = []
        self.generation_iter = 0
        self.species = [[NEAT(i, input_size, output_size) for i in range(init_population)]]
        self.comp_threshold = comp_threshold
        self.avg_comp_num = avg_comp_num
        self.mating_prob = mating_prob
        self.copy_mutate_pro = copy_mutate_pro
        self.self_mutate_pro = self_mutate_pro
        self.excess = excess
        self.disjoint = disjoint
        self.weight = weight
        self.survive = survive
        self.file_name = file_name
        self.adversarial_genomes = []

        # Load the environment parameters, if you saved it before.
        if self.file_name != None and os.path.exists(self.file_name + '.env'):
            print "Loading environment parameters...",
            self.load()
            print "\tDone!"

        for sp in self.species:
            for gen in sp:
                task.get_fitness(gen)

    def update_adversarial_genomes(self, task):
        self.adversarial_genomes = []
        for sp in self.species:
            self.adversarial_genomes.extend(sp[:])
        self.adversarial_genomes.sort(key=lambda NEAT: NEAT.fitness, reverse=True)
        self.adversarial_genomes = self.adversarial_genomes[0:(task.play_times/5)]

    def save(self):
        if self.file_name != None:
            print "Saving...",
            with open(self.file_name + '.env', "wb") as f:
                pickle.dump([ self.generation_iter,
                              self.species,
                              self.next_generation,
                              self.outcomes,
                              self.adversarial_genomes,
                              NEAT.connection_list], f)
            print "\tDone!"

    def load(self):
        if self.file_name != None:
            with open(self.file_name + '.env', "rb") as f:
                self.generation_iter, self.species, self.next_generation, self.outcomes, self.adversarial_genomes, NEAT.connection_list = pickle.load(f)

    def produce_offspring(self, genome):
        """Produce a new offspring."""
        offspring = copy.deepcopy(genome)
        offspring.id = self.evaluation
        self.next_generation.append(offspring)
        self.evaluation += 1
        return offspring

    def add_outcome(self, genome):
        """Collecting outcomes."""
        gen = copy.deepcopy(genome)
        self.outcomes.append(gen)
        # print "Generation:%d\tFound outcome %d,\thidden node = %d,\tconnections = %d"%(self.generation_iter,
        #                                                                                len(self.outcomes),
        #                                                                                len(gen.hidden_nodes),
        #                                                                                gen.connection_count())

    def mating_pair(self, pair, task):
        """Mating two genomes."""
        p1 = pair[0]
        p2 = pair[1]
        p1_len = len(p1.connections)
        p2_len = len(p2.connections)
        offspring = self.produce_offspring(NEAT(self.evaluation, self.input_size, self.output_size, offspring=True))

        # Generate the same number of nodes as the larger genome
        max_hidden_node = max(len(p1.hidden_nodes), len(p2.hidden_nodes))
        for i in range(max_hidden_node):
            offspring.add_hidden_node()

        # Crossing over
        i, j = 0, 0
        while i < p1_len or j < p2_len:
            if i < p1_len and j < p2_len:
                if p1.connections[i].innovation == p2.connections[j].innovation:
                    if NEAT.probability(0.5):
                        con = p1.connections[i]
                    else:
                        con = p2.connections[j]
                    i += 1
                    j += 1
                elif p1.connections[i].innovation < p2.connections[j].innovation:
                    con = p1.connections[i]
                    i += 1
                else:
                    con = p2.connections[j]
                    j += 1
            elif i >= p1_len:
                con = p2.connections[j]
                j += 1
            else:
                con = p1.connections[i]
                i += 1
            offspring.add_connection_id(input_node_id=con.input.id,
                                        output_node_id=con.output.id,
                                        weight=con.weight,
                                        enable=con.enable)
        # task.get_fitness(offspring)
        task.get_adversarial_fitness(offspring, self.adversarial_genomes)
        return offspring

    def mating_genomes(self, task):
        """Randomly mating two genomes."""
        mating_pool = []
        for sp in self.species:
            for gen in sp:
                # The higher the fitness, the higher the probability of mating.
                if NEAT.probability(self.mating_prob):
                    mating_pool.append(gen)

            while len(mating_pool) > 1:
                pair = random.sample(mating_pool, 2)
                self.mating_pair(pair, task)
                for p in pair:
                    mating_pool.remove(p)

    def mutation(self, task):
        """Genome mutation."""
        for k, sp in enumerate(self.species):
            for gen in self.species[k]:
                if gen.fitness < task.best_fitness:
                    if NEAT.probability(self.copy_mutate_pro):
                        offspring = self.produce_offspring(gen)
                        offspring.mutation()
                        # task.get_fitness(offspring)
                        task.get_adversarial_fitness(offspring, self.adversarial_genomes)
                    if NEAT.probability(self.self_mutate_pro):
                        gen.mutation(new_node=False)
                # task.get_fitness(gen)
                task.get_adversarial_fitness(gen, self.adversarial_genomes)

    def compatibility(self, gen1, gen2):
        """Calculating compatibility between two genomes."""
        g1_len = len(gen1.connections)
        g2_len = len(gen2.connections)
        E, D = 0, 0
        i, j = 0, 0
        w1, w2 = 0, 0
        while i < g1_len or j < g2_len:
            if i < g1_len and j < g2_len:
                if gen1.connections[i].innovation == gen2.connections[j].innovation:
                    w1 += abs(gen1.connections[i].weight)
                    w2 += abs(gen2.connections[j].weight)
                    i += 1
                    j += 1
                elif gen1.connections[i].innovation < gen2.connections[j].innovation:
                    D += 1
                    w1 += abs(gen1.connections[i].weight)
                    i += 1
                else:
                    D += 1
                    w2 += abs(gen2.connections[j].weight)
                    j += 1
            elif i >= g1_len:
                E += 1
                w2 += abs(gen2.connections[j].weight)
                j += 1
            else:
                E += 1
                w2 += abs(gen1.connections[i].weight)
                i += 1
        W = abs(w1 - w2)
        distance = self.excess * E + self.disjoint * D + self.weight * W
        return distance

    def speciation(self, genome):
        """Assign a genome to compatible species."""
        for sp in self.species:
            avg_comp = 0
            for gen in sp[:self.avg_comp_num]:
                avg_comp += self.compatibility(gen, genome)
            avg_comp /= min(self.avg_comp_num, len(sp))
            if avg_comp < self.comp_threshold:
                sp.append(genome)
                return
        # If there is no compatible species, create a new species for the genome.
        if len(self.species) < 15:
            self.species.append([genome])

    def surviving_rule(self):
        """Set the surviving rules."""

        for gen in self.next_generation:
            self.speciation(gen)

        for k, sp in enumerate(self.species):
            sp.sort(key=lambda NEAT: NEAT.fitness, reverse=True)
            # sp = sp[:20] + self.next_generation + [NEAT(i,
            #                                             self.input_size,
            #                                             self.output_size)
            #                                        for i in range(10)]
            self.species[k] = self.species[k][:self.survive]

    def run(self, task, showResult=False):
        """Run the environment."""
        print "Running Environment...(Initial population = %d, Maximum generation = %d)"%(self.population, self.max_generation)
        # Generational change
        for self.generation_iter in range(self.generation_iter + 1, self.max_generation):
            self.next_generation = []

            # Mutation
            self.mutation(task)

            # Mating genomes
            self.mating_genomes(task)

            # Killing bad genomes
            self.surviving_rule()

            # Logging outcome information
            outcome = [gen for sp in self.species for gen in sp if gen.fitness >= task.best_fitness]
            self.population = sum([len(sp) for sp in self.species])
            hidden_distribution = [0]
            max_fitness = -sys.maxint
            best_outcome = None
            for sp in self.species:
                genome_len = len(sp)
                if genome_len > 0:
                    for gen in sp:
                        hid = len(gen.hidden_nodes)
                        while hid >= len(hidden_distribution):
                            hidden_distribution.append(0)
                        hidden_distribution[hid] += 1
                        if gen.fitness > max_fitness:
                            max_fitness = gen.fitness
                            best_outcome = gen

            print "Generation %d:\tpopulation = %d,\tspecies = %d,\toutcome = %d,\tbest_fitness(%d_%d) = %.2f%%,\thidden node distribution:%s"%(
                self.generation_iter,
                self.population,
                len(self.species),
                len(outcome),
                best_outcome.id,
                len(best_outcome.hidden_nodes),
                100.0 * max_fitness / task.best_fitness,
                hidden_distribution)

            # Update adversarial genomes
            # if self.generation_iter >= 100 and self.generation_iter % 10 == 0:
            #     self.update_adversarial_genomes(task)
            #     print "Adversarial genomes updated."

            # Save genome
            if self.file_name != None:
                with open(self.file_name + '.gen', 'wb') as file_out:
                    pickle.dump(best_outcome, file_out)

            # Save environment parameters
            if self.generation_iter % 10 == 0:
                self.save()

        # Collecting outcomes
        max_fitness = -sys.maxint
        best_outcome = None
        for sp in self.species:
            for gen in sp:
                if gen.fitness >= task.best_fitness:
                    self.add_outcome(gen)
                if gen.fitness > max_fitness:
                    max_fitness = gen.fitness
                    best_outcome = gen
        # if len(self.outcomes) == 0:
        #     self.add_outcome(best_outcome)

        # Save best genome
        if self.file_name != None:
            with open(self.file_name + '.gen', 'wb') as file_out:
                pickle.dump(best_outcome, file_out)

        print "Species distribution:"
        for k, sp in enumerate(self.species):
            hidden_node = []
            con = []
            for gen in sp:
                hidden_node.append(len(gen.hidden_nodes))
                con.append(len(gen.connections))
            print "\t%d:\tnode:\t%s\n\t\tcons:\t%s"%(k, hidden_node, con)
        print

        if showResult:
            print "Completed Genomes:",
            self.outcomes.sort(key=lambda NEAT:NEAT.hidden_nodes)
            outcomes_len = len(self.outcomes)
            if outcomes_len > 0:
                print outcomes_len
            else:
                print "There are no completed genomes!"
            avg_hid = 0.0
            avg_con = 0.0
            if outcomes_len > 0:
                for gen in self.outcomes:
                    gen.show_structure()
                    avg_hid += len(gen.hidden_nodes)
                    avg_con += gen.connection_count()
                avg_hid /= outcomes_len
                avg_con /= outcomes_len
            print "Evaluation: %d,\tPopulation: %d,\tAverage Hidden node = %f,\tAverage Connection = %f"%(
self.evaluation, self.population, avg_hid, avg_con)


## Connection.py
import random

class Connection(object):
    """The connection of evolutionary neural network.
    Attributes:
        input: The input node of the connection.
        output: The output node of the connection.
        weight: The weight of the connection.
        enable: The enable flag for the connection
        innovation: The number of innovation for the connection.
    """
    def __init__(self, input, output, weight = None, innovation = None, enable = True):
        self.input = input
        self.output = output
        if weight != None:
            self.weight = weight
        else:
            self.weight = self.random_weight()
        self.enable = enable
        self.innovation = innovation

    def __eq__(self, other):
        """If the input node and the output node are the same in both connections,
        then the two connections are considered the same.
        """
        return self.input == other.input and self.output == other.output

    def random_weight(self, range = 10):
        """Randomly generate the weight of the connection. The default range is from -10 to 10."""
        self.weight = random.uniform(-range, range)
