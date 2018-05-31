##TODO:

## Environment
## Genome
## Pool

##DONE:
## 1. Gene.
## 2. Node.
## 3. Species.

import copy
import math
import random

from decimal import Decimal

class Activations:
    ## TODO: Documentation.
    def __init__(self, default="SIGMOID"):
        self.default = default

    def base_sigmoid(self, input_val):
        ## This is just calculating a sigmoid:
        ## y = 1/(1 + e^(-ax))
        power = Decimal(-4.9 * input_val)
        #print(power)
        bottom = Decimal(1.0) + Decimal.exp(power)
        result = Decimal(1.0) / bottom
        result = float(result)
        return result

class Network:
    ## TODO: Documentation.
    def __init__(self):
        pass

class Neuron:
    ## TODO: Documentation
    def __init__(self, weight, bias, value=1.0):
        ## TODO: Make sure to add in an option for neurons that build
        ## until they hit a threshold then activate and reset.
        self.weight = weight
        self.bias = bias
        self.value = value
        self.last_value = value

class Species:
    ## TODO: Documentation.
    def __init__(self, initial_genome, species_number=0):
        ## TODO: Should we add special growth attributes to each
        ## species so that we get even larger differences?
        ## If a new species is being made then there is a genome
        ## that doesn't fit with any of the other species. This means
        ## a new species will always start with one genome.
        self.genomes = [initial_genome]
        self.current_size = 1
        ## The first genome will always be the representative of this
        ## species.
        self.representative = self.genomes[0]
        ## This is used to make sure that interspecies breeding
        ## doesn't accidentally breed the same species.
        self.species_number = species_number
        ## Used to compare species.
        self.adjusted_fitness_sum = 0.0
        ## Used to see if there has been any recent progress.
        self.maximum_adjusted_fitness = 0.0
        ## How many generations it has been since progress.
        self.stagnant_time = 0
        ## The index of the current best genome.
        self.current_max_genome = 0

    def calculate_adjusted_fitness(self):
        ## Used to calculate the adjusted fitness of the
        adjusted_sum = 0.0

        max_adjusted_fitness = 0
        max_key = 0
        ## Find the adjusted fitness of each genome then sum them.
        for key, genome in enumerate(self.genomes):
            genome.calculate_adjusted_fitness(self.current_size)
            if genome.adjusted_fitness > max_adjusted_fitness:
                max_adjusted_fitness = genome.adjusted_fitness
                max_key = key
            adjusted_sum += genome.adjusted_fitness
        self.current_max_genome = max_key
        self.adjusted_fitness_sum = adjusted_sum
        #print("Adjusted fitness sum: ", self.adjusted_fitness_sum)
        #input("...")

        ## Check if the species made progress this generation.
        if self.adjusted_fitness_sum > self.maximum_adjusted_fitness + 0.05 * self.maximum_adjusted_fitness:
            self.stagnant_time = 0
        else:
            self.stagnant_time += 1

    def remove_weak_genomes(self, all_but_one=False):
        self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
        '''
        genome_fitness_data = []
        for key, genome in enumerate(self.genomes):
            genome_fitness_data.append({'key':key, 'fitness':genome.fitness})
        genome_fitness_data.sort(key=lambda data: data['fitness'])
        '''
        if not all_but_one and len(self.genomes) > 1:
            num_to_keep = math.ceil(len(self.genomes)//2)
        else:
            num_to_keep = 1
        self.genomes = self.genomes[:num_to_keep]

        '''
        genome_fitness_data = genome_fitness_data[:num_to_keep]
        '''
        '''
        self.genomes.sort(key=lambda genome: genome.adjusted_fitness, reverse=False)
        if not all_but_one:
            num_to_keep = math.ceil(len(self.genomes)//2)
        else:
            num_to_keep = 1

        self.genomes = self.genomes[:num_to_keep]
        '''

        #return genome_fitness_data



class Genome:
    ## TODO: Documentation.
    def __init__(self, num_inputs=0, num_outputs=0, genes=None,
                 neuron_input_nodes=None, neuron_output_nodes=None,
                 neuron_hidden_nodes=None, fitness=0.0, species=None,
                 genome_number=0):
        ## TODO: Add some documentation here.

        self.network_created = False

        self.activations = Activations()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if genes is not None:
            self.genes = genes
        else:
            self.genes = []
        if neuron_input_nodes is not None:
            self.neuron_input_nodes = neuron_input_nodes
        else:
            self.neuron_input_nodes = []
        if neuron_output_nodes is not None:
            self.neuron_output_nodes = neuron_output_nodes
        else:
            self.neuron_output_nodes = []
        if neuron_input_nodes is not None:
            self.neuron_input_nodes = neuron_input_nodes
        else:
            for i in range(num_inputs):
                self.neuron_input_nodes.append(Node())
        if neuron_output_nodes is not None:
            self.neuron_output_nodes = neuron_output_nodes
        else:
            for i in range(num_outputs):
                self.neuron_output_nodes.append(Node())

        if neuron_hidden_nodes is not None:
            self.neuron_hidden_nodes = neuron_hidden_nodes
        else:
            self.neuron_hidden_nodes = []
        self.num_hidden_nodes = len(self.neuron_hidden_nodes)


        self.fitness = fitness
        self.adjusted_fitness = fitness
        self.genome_number = genome_number


        self.active_gene_set = set()
        self.active_gene_dict = {}

        self.links = set()

    def add_gene(self, gene):
        ## Add a new gene to the genome.
        temp_len = len(self.genes)
        self.active_gene_set.add(gene.innovation)
        self.active_gene_dict[gene.innovation] = temp_len
        self.genes.append(gene)

    def sort_genes(self):
        ## Sort genes within the genome for crossover analysis.
        self.genes = sorted(self.genes, key=lambda gene: gene.innovation)

    def calculate_adjusted_fitness(self, species_size):
        ## The adjusted fitness is the fitness of the genome divided
        ## by the number of members in the species it belongs to.
        self.adjusted_fitness = self.fitness / float(species_size)
        #print("Genome fitness: ", self.fitness, self.adjusted_fitness, species_size)

    def create_network(self):
        hidden_node_set = set()
        current_hidden_node_counter = 0
        self.hidden_node_dict = {}
        for gene in self.genes:
            #print("Gene in_node:{}, in_type:{}, out_node:{}, out_type:{}, weight:{}, bias:{}".format(gene.in_node, gene.input_type, gene.out_node, gene.output_type, gene.weight, gene.innovation))

            if gene.output_type == "HIDDEN":

                if gene.out_node not in hidden_node_set:
                    self.hidden_node_dict[gene.out_node] = current_hidden_node_counter
                    current_hidden_node_counter += 1
                hidden_node_set.add(gene.out_node)
        self.num_hidden_nodes = current_hidden_node_counter

        for key, temp_node_number in enumerate(list(hidden_node_set)):
            self.hidden_node_dict[temp_node_number] = key


        self.num_hidden_nodes = len(hidden_node_set)

        self.neuron_hidden_nodes = [Node() for _ in range(self.num_hidden_nodes)]
        self.neuron_output_nodes = [Node() for _ in range(self.num_outputs)]
        for gene in self.genes:
            if gene.output_type == "HIDDEN":
                self.neuron_hidden_nodes[self.hidden_node_dict[gene.out_node]].input_connections.add(gene)
            if gene.output_type == "OUTPUT":
                self.neuron_output_nodes[gene.out_node].input_connections.add(gene)
        self.network_created = True

    def run_network(self, inputs):
        if not self.network_created:
            #print("Creation test.")
            self.sort_genes()
            self.create_network()
        for i in range(self.num_inputs - 1):
            self.neuron_input_nodes[i].value = inputs[i]
        self.neuron_input_nodes[-1].value = 1.0

        for node in self.neuron_hidden_nodes:
            temp_sum = 0.0
            for connection in node.input_connections:
                if connection.enabled:
                    temp_in_node = connection.in_node
                    if connection.input_type == "INPUT":
                        temp_in_value = self.neuron_input_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = 0
                        #temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        temp_sum += temp_value
                    elif connection.input_type == "HIDDEN" and temp_in_node in self.hidden_node_dict:
                        temp_in_value = self.neuron_hidden_nodes[self.hidden_node_dict[temp_in_node]].value
                        temp_weight = connection.weight
                        temp_bias = 0
                        #temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        temp_sum += temp_value
                    elif connection.input_type == "OUTPUT":
                        temp_in_value = self.neuron_output_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = 0
                        #temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        temp_sum += temp_value

            node.value = self.activations.base_sigmoid(temp_sum)

        output = [0.0] * self.num_outputs
        for key, node in enumerate(self.neuron_output_nodes):
            temp_sum = 0.0
            #print(node.input_connections)
            #input('....')
            for connection in node.input_connections:
                if connection.enabled:
                    temp_in_node = connection.in_node
                    if connection.input_type == "INPUT":
                        temp_in_value = self.neuron_input_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        #print("Current node:", key)
                        #print("connection",connection.printable())
                        #print(temp_value)
                        #print("weight", temp_weight)
                        temp_sum += temp_value
                        #print("temp_sum", temp_sum)
                    elif connection.input_type == "HIDDEN" and temp_in_node in self.hidden_node_dict:
                        #print("Temp in node: {}".format(temp_in_node))
                        #print(self.hidden_node_dict)
                        temp_in_value = self.neuron_hidden_nodes[self.hidden_node_dict[temp_in_node]].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        temp_sum += temp_value
                    elif connection.input_type == "OUTPUT":
                        temp_in_value = self.neuron_output_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_value = temp_in_value * temp_weight# + temp_bias
                        temp_sum += temp_value

            node.value = self.activations.base_sigmoid(temp_sum)
            if len(node.input_connections) == 0:
                node.value = 0.0
            #print("node_value",node.value)
            #input("...")
            output[key] = node.value
        #print("output", output)
        return output


class Gene:
    ## TODO: Documentation.
    def __init__(self, in_node=None, input_type=None,
                 out_node=None, output_type=None, weight=None,
                 bias=None, initial_value=0.0,
                 enabled=False, innovation=0,
                 recurrent=False):
        ## Used to calculate this nodes output.
        self.in_node = in_node
        ## Can be INPUT or HIDDEN or OUTPUT
        self.input_type = input_type
        ## Used when adding in new nodes between this node and its output.
        self.out_node = out_node
        ## Can be HIDDEN or OUTPUT or OUTPUT
        self.output_type = output_type
        ## This is multiplied by the inputs.
        self.weight = weight
        ## This is added to the calculated value.
        self.bias = bias
        ## This is what the neuron should initially be set to.
        self.initial_value = initial_value
        ## If this node should be added to the tree.
        self.enabled = enabled
        ## Global innovation number for tracking unique changes to the
        ## genome.
        self.innovation = innovation
        ## Used to check for recurrence.
        self.recurrent = recurrent

    def printable(self):
        return "into:{}, in_type{}, output:{}, out_type:{} weight:{}, bias:{}, enabled:{}".format(
            self.in_node, self.input_type, self.out_node, self.output_type, self.weight, self.bias, self.enabled
        )

class Node:
    ## TODO: Documentation.
    def __init__(self, input_connections=None, output_connections=None,
                 hidden_connections=None, value=0.0):
        if input_connections is not None:
            self.input_connections = input_connections
        else:
            self.input_connections = set()
        if output_connections is not None:
            self.output_connections = output_connections
        else:
            self.output_connections = set()
        if hidden_connections is not None:
            self.hidden_connections = hidden_connections
        else:
            self.hidden_connections = set()
        self.value = value

class Population:
    ## TODO: Documentation.
    def __init__(self, num_inputs, num_outputs, seed=None, frames_to_update=300,
                 population_size=300, max_num_species=15):
        ## Generator with seed. If a seed is given it should be
        ## deterministic.
        self.generator = random.Random()
        if seed:
            self.seed = seed
            self.generator = random.Random()
            self.generator.seed(seed)
        else:
            self.seed = None
            self.generator = random.Random()


        ## Set the population size.
        self.population_size = population_size
        ## Constants used to separate genomes into species.
        ## Constant for excess genes.
        self.c1 = 1.0
        ## Constant for disjoint genes.
        self.c2 = 1.0
        ## By increasing this the system accounts for more variations
        ## in gene weights
        self.c3 = 0.3 ## 3.0 for large populations

        ## Constant for acceptable degree of differentiation between
        ## species
        self.del_t = 0.3 ## 4.0 for large populations

        ## If species have been stagnant for this amount of time
        ## that species is no longer allowed to reproduce. Stagnant
        ## is defined by no increase in fitness.
        self.stagnant_time = 15

        ## Number of members of each species to copy champion with
        ## no changes.
        ## TODO: Actually use this.
        self.champ_species_size = 5

        ## Chance a genome will have weights manipulated.
        self.weight_manipulation_chance = 0.8

        ## This is used for deciding perturbation or full random changes.
        ## This is only used if the weight manipulation check passes.
        ## 90% perturbation chance, 10% random assignment chance.
        self.weight_perturbation_chance = 0.9

        ## If the gene is disabled in either parent there is this chance
        ## that the child will also have the gene disabled.
        self.disable_chance = 0.95

        ## This is the chance that the genome will mutate without any
        ## crossover.
        self.no_crossover_chance = 0.25

        ## The chance that a genome will mate with a unit from a
        ## different species.
        self.inter_species_chance = 0.001

        ## The chance a new node will be added to a genome.
        self.node_generation_chance = 0.03

        ## The chance a new link will be added to a genome. For large
        ## populations this can be increased to 0.3. Link generation
        ## was found to be more important than node generation in the
        ## original paper.
        self.link_generation_chance = 0.05

        self.link_deletion_chance = 0.05

        ## TODO: Should we add in the ability to do contraction
        ## manipulations? We would need something for keeping track
        ## of how long we have been on each phase of generation and
        ## culling, how well we have been progressing recently,
        ## and which stage we are on.

        ## The maximum amount a mutation can perturb a weight.
        self.max_weight_perturbation = 0.5
        ## The maximum amount a mutation can perturb a bias.
        self.max_bias_perturbation = 1.0

        ## The maximum value of a replaced weight.
        self.max_weight_replaced = 1.0
        ## The maximum value of a replaced bias.
        self.max_bias_replaced = 0.5

        ## Max number of frames to use per network run.
        self.frames_to_update = frames_to_update

        ## Number of inputs to the network.
        self.num_inputs = num_inputs + 1

        ## Number of outputs for the network.
        self.num_outputs = num_outputs

        ## For testing. TODO: remove this eventually.
        self.max_num_genes = 0


        self.max_num_species = max_num_species


    def setup(self, previous_paths=None):

        if previous_paths is None:
            self.previous_path = []
        else:
            self.previous_path = previous_path
        self.current_position = [0.0, 0.0]

        ## This is used to track each new innovation in a generation.
        ## If it has been found before, give both genes the same innovation
        ## number. Each gene will be given as
        ## {(INPUT_TYPE, INPUT_VALUE, OUTPUT_TYPE, OUTPUT_VALUE): innovation}
        self.generation_innovations = {}
        ## This was changed.
        ## I think this is right? the basic genome should have
        ## num_inputs * num_outputs genes so new innovations should
        ## start past there.
        ##self.global_innovation = num_inputs * num_outputs
        ## Start at innovation zero.
        self.global_innovation = 0

        ## List for keeping each species in.
        self.species_list = []

        ## Used for finding the number of each species to be created.
        self.total_adjusted_fitness = 0.0

        ## Current generation.
        self.current_generation = 0

        ## Highest fitness found.
        self.highest_fitness = 0.0

        ## Current species being tested.
        self.current_species = 0

        ## Current genome being tested.
        self.current_genome = 0

        ## Number of frames that have been processed with a network.
        self.current_frame = 0


        self.species_counter = 0

        self.genome_counter = 0

        ## Used to check if the current generation has been fully processed.
        self.generation_processed = False

        ## Used to check if the current genome has been fully processed.
        self.genome_processed = False
        self.species_processed = False
        self.generation_processed = False


        self.best_genome = None

        self.extra_frames_earned = 0
        #########################


        err = self.generate_initial_population()
        self.generation_processed = False
        if err is not None:
            print("Error setting up population. {}".format(err))

    def new_innovation(self):
        temp_innovation = self.global_innovation
        self.global_innovation += 1
        return temp_innovation

    def calculate_compatibility(self, genome_1, genome_2):
        ## Used to calculate if two genomes should be in the same species.
        ## Sort both genomes for simplicity. Returns a pair of
        ## (compatability_distance, error)
        if len(genome_1.genes) < 1:
            return (None, 1)
        if len(genome_2.genes) < 1:
            return (None, 2)
        genome_1.sort_genes()
        genome_2.sort_genes()
        ## Calculate the variables necissary to calculate compatability.
        excess, disjoint, N, W = self.calculate_excess_and_disjoint(genome_1, genome_2)
        #print("excess:{} disjoint:{} N:{} W:{}".format(excess, disjoint, N, W))
        ## Calculate compatability between two genes and return it.
        if N > 0:
            compatability_distance = (self.c1 * excess + self.c2 * disjoint) / N# + self.c3 * W
        else:
            compatability_distance = (self.c1 * excess + self.c2 * disjoint)# + self.c3 * W

        return (compatability_distance, None)

    def calculate_excess_and_disjoint(self, genome_1, genome_2):
        ## We can assume here that there will be at least one gene
        ## in both genomes.
        ## Find the index of the last gene in each genome.
        genome_len_1 = len(genome_1.genes) - 1
        genome_len_2 = len(genome_2.genes) - 1
        ## Find the length of the longer genome.
        if genome_len_1 >= genome_len_2:
            n = genome_len_1 + 1
        else:
            n = genome_len_2 + 1

        ## These are used to keep track of the current position in each
        ## genome.
        current_position_1 = 0
        current_position_2 = 0
        ## Keep track of the number of excess and disjoint genes
        ## to return.
        excess = 0
        disjoint = 0
        ## Keep track of the differences of weights in matching genes.
        ## Possible TODO: Keep track of biases/start value as well?
        weight_differences_sum = 0
        num_matching_genes = 0
        ## Used to mark if the process is done.
        not_done = True
        while not_done:
            ## If we are at the end of both. Stop.
            #print("Genome 1 gene:{}, genome 1 innovation:{}, genome 2 gene:{}, genome 2 innovation:{}".format(current_position_1, genome_1.genes[current_position_1].innovation, current_position_2, genome_2.genes[current_position_2].innovation))
            if current_position_1 == genome_len_1 and current_position_2 == genome_len_2:
                ## If both gene postions are the final positions we are
                ## done.
                not_done = False
                ## In this case if the current genemoe innovation is not
                ## the same it means we have one more excess gene.
                if genome_1.genes[current_position_1].innovation != genome_2.genes[current_position_2].innovation:
                    excess += 1
                else:
                    ## Otherwise we have one more matching gene.
                    num_matching_genes += 1
                    weight_differences_sum += \
                        abs(genome_1.genes[current_position_1].weight +\
                        genome_2.genes[current_position_2].weight)
            elif (genome_1.genes[current_position_1].innovation ==
            genome_2.genes[current_position_2].innovation):
                ## If the current innovations are the same we need to
                ## add that information to the weight differences and
                ## matching variables.
                weight_differences_sum += \
                    abs(genome_1.genes[current_position_1].weight +\
                    genome_2.genes[current_position_2].weight)
                num_matching_genes += 1
                ## Check on each genome if it is at the end. If not
                ## move the current marker one down the genome.
                if current_position_1 != genome_len_1:
                    current_position_1 += 1
                if current_position_2 != genome_len_2:
                    current_position_2 += 1

            elif (genome_1.genes[current_position_1].innovation >
            genome_2.genes[current_position_2].innovation):
                ## If we are already at the end of genome 2 increment
                ## the place in genome 2 and say the current comparison
                ## is an excess.
                ## Otherwise, it is a disjoint gene so increment the
                ## position in the second genome to try and find a gene
                ## with a higher innovation.
                if current_position_2 == genome_len_2:
                    excess += 1
                    current_position_1 += 1
                else:
                    disjoint += 1
                    current_position_2 += 1


            elif (genome_1.genes[current_position_1].innovation <
            genome_2.genes[current_position_2].innovation):
                ## If we are already at the end of genome 1 increment
                ## the place in genome 2 and say the current comparison
                ## is an excess.
                ## Otherwise, it is a disjoint gene so increment the
                ## position in the first genome to try and find a gene
                ## with a higher innovation.
                if current_position_1 == genome_len_1:
                    excess += 1
                    current_position_2 += 1
                else:
                    disjoint += 1
                    current_position_1 += 1

        w = float(weight_differences_sum)
        return (excess, disjoint, n, w)

    def add_genome_to_species(self, genome):
        ## Returns error.
        ## TODO: Make sure to delete species if they don't have any genomes.num_input
        for species in self.species_list:
            ##print("Type of representative: {}".format(type(species.representative)))
            compatability, err = self.calculate_compatibility(species.representative, genome)
            if err is not None:
                print(err)
                return err
            else:

                #print("compatability: {}.".format(compatability))
                if compatability < self.del_t:
                    #print("compatability: {}.".format(compatability))
                    species.genomes.append(genome)
                    species.current_size += 1
                    return None
        ## If we were unable to find one. Create a new species and add it.
        ##print("Genome type: {}".format(type(genome)))
        self.species_list.append(Species(genome, self.species_counter))
        self.species_counter += 1
        return None

    def generate_initial_population(self):
        ## Returns error.
        for i in range(self.population_size):
            temp_genome = self.starter_genome()
            ## TODO: This is for testing, remove later.
            for gene in temp_genome.genes:
                #print("Temp_Gene2 in_node:{}, in_type:{}, out_node:{}, out_type:{}, weight:{}, bias:{}".format(gene.in_node, gene.input_type, gene.out_node, gene.output_type, gene.weight, temp_genome.genome_number))
                pass
            ##
            err = self.add_genome_to_species(temp_genome)

        total_num_genomes = 0
        for species in self.species_list:
            total_num_genomes += len(species.genomes)
        #print("1Total number of genomes: {}\nNumber of species: {}".format(total_num_genomes, len(self.species_list)))

        return err

    def starter_genome(self, num_input_connections=5, num_output_connections=5):
        ## Returns a new basic genome.
        ## Make a new genome with the right number of inputs and outputs.
        new_genome = Genome(self.num_inputs, self.num_outputs, [], genome_number=self.genome_counter)
        ##print("Genome type: {}".format(type(new_genome)))
        self.genome_counter += 1

        #print(self.generation_innovations)

        '''
        ## Each basic genome will have the same innovations to fully
        ## connect them.
        temp_innovations = 0

        ## Fully connect the network.
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                ## Random weight and bias information.
                temp_weight = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
                temp_bias = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
                temp_gene = Gene(i, "INPUT", j, "OUTPUT", temp_weight,
                                 temp_bias, enabled=True,
                                 temp_innovations)
                ## This will be used in node mutation later.
                temp_len = len(new_genome.genes)
                new_genome.active_gene_dict[temp_innovations] = temp_len
                new_genome.active_gene_set.add(temp_innovations)
                new_genome.genes.append(temp_gene)
                temp_innovations += 1
        '''

        ## Connect five random input nodes to five random output nodes.
        if num_input_connections > self.num_inputs:
            num_input_connections = self.num_inputs
        if num_output_connections > self.num_outputs:
            num_output_connections = self.num_outputs

        input_values = self.generator.sample(range(0, self.num_inputs), num_input_connections)
        #print(self.num_inputs)
        #print(input_values)
        #input("...")
        output_values = self.generator.sample(range(0, self.num_outputs), num_output_connections)

        input_type = "INPUT"
        output_type = "OUTPUT"

        for input_node in input_values:
            for output_node in output_values:
                #print("INPUT", input_node, "OUTPUT", output_node)
                ## Dealing with the innovation.
                if ("INPUT", input_node, "OUTPUT", output_node) in self.generation_innovations:
                    temp_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
                    new_genome.links.add((input_type, input_node, output_type, output_node))
                else:
                    temp_innovation = self.new_innovation()
                    self.generation_innovations[(input_type, input_node, output_type, output_node)] = temp_innovation
                    new_genome.links.add((input_type, input_node, output_type, output_node))

                ## Random weight and bias information.
                temp_weight = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
                #temp_bias = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
                temp_bias = 0.0
                temp_gene = Gene(input_node, "INPUT", output_node, "OUTPUT", temp_weight,
                                 temp_bias, enabled=True,
                                 innovation=temp_innovation)
                ## This will be used in node mutation later.
                temp_len = len(new_genome.genes)
                new_genome.active_gene_dict[temp_innovation] = temp_len
                new_genome.active_gene_set.add(temp_innovation)
                new_genome.genes.append(temp_gene)
        #print("new_genome",new_genome.genome_number,len(new_genome.genes))
        #for temp_gene in new_genome.genes:
        #    print("Temp_Gene in_node:{}, in_type:{}, out_node:{}, out_type:{}, weight:{}, bias:{}".format(temp_gene.in_node, temp_gene.input_type, temp_gene.out_node, temp_gene.output_type, temp_gene.weight, new_genome.genome_number))

        new_genome.num_hidden_nodes = 0


        return new_genome

    def breed_genomes(self, genome_1, genome_2):
        ## The fitness of each genome.
        fitness_1 = genome_1.fitness
        fitness_2 = genome_2.fitness
        ## Used for keeping track of genes.
        genome_1_dict = {}
        genome_2_dict = {}
        ## Collect innovations and their associated gene in the dicts.
        for key, gene in enumerate(genome_1.genes):
            genome_1_dict[gene.innovation] = key
        for key, gene in enumerate(genome_2.genes):
            genome_2_dict[gene.innovation] = key

        ## Genome to return.
        new_genome = Genome(self.num_inputs, self.num_outputs)
        if fitness_1 > fitness_2:
            ## Fitness one is greater.
            ## Step through genome_1's genes.
            for gene in genome_1.genes:
                ## For matching genes, randomly choose one to inherit.
                if gene.innovation in genome_2_dict:
                    ## In both genomes.
                    random_guess = self.generator.random()
                    if random_guess < 0.5:
                        ## Take from the first.
                        new_gene = copy.deepcopy(gene)
                    else:
                        ## Take from the second.
                        new_gene = copy.deepcopy(genome_2.genes[genome_2_dict[gene.innovation]])

                ## Copy the disjoint and excess genes from genome_1.
                else:
                    new_gene = copy.deepcopy(gene)

                ## If genome is disabled in either parent there is a
                ## chance of reviving it in the child.
                if not new_gene.enabled:
                    random_guess = self.generator.random()
                    if random_guess > self.disable_chance:
                        new_gene.enabled = True
                new_genome.genes.append(new_gene)

        elif fitness_2 > fitness_1:
            ## Fitness two is greater.
            ## Step through genome_2's genes.
            for gene in genome_2.genes:
                ## For matching genes, randomly choose one to inherit.
                if gene.innovation in genome_1_dict:
                    ## In both genomes.
                    random_guess = self.generator.random()
                    if random_guess < 0.5:
                        ## Take from the second.
                        new_gene = copy.deepcopy(gene)
                    else:
                        ## Take from the first.
                        new_gene = copy.deepcopy(genome_1.genes[genome_1_dict[gene.innovation]])

                ## Copy the disjoint and excess genes from genome_2.
                else:
                    new_gene = copy.deepcopy(gene)

                ## If genome is disabled in either parent there is a
                ## chance of reviving it in the child.
                if not new_gene.enabled:
                    random_guess = self.generator.random()
                    if random_guess > self.disable_chance:
                        new_gene.enabled = True
                new_genome.genes.append(new_gene)

        else:
            ## Both fitnesses are the same.
            ## Keep dijoint and excess from both.
            ## Step through each set of genes.
            for gene in genome_1.genes:
                new_gene = None
                ## For matching genes, randomly choose one to inherit.
                if gene.innovation in genome_2_dict:
                    ## In both genomes.
                    random_guess = self.generator.random()
                    if random_guess < 0.5:
                        ## Take from the first.
                        new_gene = copy.deepcopy(gene)
                    else:
                        ## Take from the second.
                        new_gene = copy.deepcopy(genome_2.genes[genome_2_dict[gene.innovation]])
                    del genome_2_dict[gene.innovation]

                else:
                    ## Always keep it.
                    new_gene = copy.deepcopy(gene)
                ## Check if there is a new gene to add.
                if new_gene:
                    ## Check if it is disabled and should be enabled.
                    if not new_gene.enabled:
                        random_guess = self.generator.random()
                        if random_guess > self.disable_chance:
                            new_gene.enabled = True
                    new_genome.genes.append(new_gene)

            for gene in genome_2.genes:
                if gene.innovation in genome_2_dict:
                    ## It is a disjoint or excess gene in genome 2.
                    ## Add it to the new genome.
                    new_gene = copy.deepcopy(gene)
                    ## Check if it is disabled and should be enabled.
                    if not new_gene.enabled:
                        random_guess = self.generator.random()
                        if random_guess > self.disable_chance:
                            new_gene.enabled = True
                    new_genome.genes.append(new_gene)
        for key, gene in enumerate(new_genome.genes):
            if gene.enabled:
                new_genome.active_gene_dict[gene.innovation] = key
                new_genome.active_gene_set.add(gene.innovation)

        ## Add in the gene keys to the new genome.
        for gene in new_genome.genes:
            input_type = gene.input_type
            input_node = gene.in_node
            output_type = gene.output_type
            output_node = gene.out_node
            new_genome.links.add((input_type, input_node, output_type, output_node))

        return new_genome

    def delete_link(self, genome):
        random_index = self.generator.randrange(0, len(genome.genes))
        genome.genes[random_index].enabled = False


    def mutate_link(self, genome):
        ## TODO: Finish documenting.
        ## This does not add any new nodes into the network.
        ## Connect two random nodes to each other.
        ## Check if these two are connected already.
        ## Only output to hidden and output nodes.
        #print("num nodes: {}".format(genome.num_inputs, genome.num_outputs, genome.num_hidden_nodes))
        random_index = self.generator.randrange(0, genome.num_inputs + genome.num_outputs + genome.num_hidden_nodes)
        if random_index < genome.num_inputs:
            ## Use an input as the input node.
            input_node = random_index
            input_type = "INPUT"
            pass
        elif random_index < genome.num_inputs + genome.num_outputs:
            ## Use an output as the input node.
            input_node = random_index - genome.num_inputs
            input_type = "OUTPUT"
        else:
            ## Use a hidden node as the input node.
            input_node = random_index - genome.num_inputs - genome.num_outputs
            input_type = "HIDDEN"

        random_index = self.generator.randrange(0, genome.num_outputs + genome.num_hidden_nodes)
        if random_index < genome.num_outputs:
            ## Use an output as the input node.
            output_node = random_index
            output_type = "OUTPUT"
        else:
            ## Use a hidden node as the input node.
            output_node = random_index - genome.num_outputs
            output_type = "HIDDEN"

        if (input_type, input_node, output_type, output_node) not in genome.links:
            #print("Links: ",genome.links)
            if (input_type, input_node, output_type, output_node) in self.generation_innovations:
                temp_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
            else:
                temp_innovation = self.new_innovation()
                self.generation_innovations[(input_type, input_node, output_type, output_node)] = temp_innovation
                genome.links.add((input_type, input_node, output_type, output_node))
            if input_type == output_type and input_node == output_node:
                temp_reccurent = True
            else:
                temp_reccurent = False
            temp_weight = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
            #temp_bias = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
            temp_bias = 0.0
            temp_len = len(genome.genes)
            genome.active_gene_dict[temp_innovation] = temp_len
            genome.active_gene_set.add(temp_innovation)
            genome.genes.append(Gene(input_node, input_type,
                                     output_node, output_type,
                                     temp_weight, temp_bias, enabled=True,
                                     innovation=temp_innovation,
                                     recurrent=temp_reccurent))

    def mutate_node(self, genome):
        ## Take one current gene. Dissable it. Create two new genes
        ## the first of which goes from the current gene's input to
        ## a new node, the second goes from the new node to the
        ## current gene's output.
        ## TODO: Document better.
        genome_length = len(genome.genes)

        ## Check if there is even a link to split with a node.
        if genome_length > 0:
            ## Our current gene.
            random_index = self.generator.randrange(0, len(genome.active_gene_set))
            gene_innovation = list(genome.active_gene_set)[random_index]
            #print("gene_inovation", gene_innovation)
            #print(genome.active_gene_dict)
            #print(genome.genes)
            old_gene = genome.genes[genome.active_gene_dict[gene_innovation]]
            genome.active_gene_set.remove(gene_innovation)
            del genome.active_gene_dict[gene_innovation]
            ## Dissable this gene.
            old_gene.enabled = False
            ## Create two new genes using the input and output of the current
            ## gene.
            ## This deals with the current input.
            old_input_type = old_gene.input_type
            old_input_node = old_gene.in_node
            old_output_type = old_gene.output_type
            old_output_node = old_gene.out_node

            temp_recurrent = False

            new_output_type_1 = "HIDDEN"
            new_input_type_2 = "HIDDEN"

            new_output_node = len(genome.genes)
            new_input_node = len(genome.genes)

            if (old_input_type, old_input_node, new_output_type_1, new_output_node) in self.generation_innovations:
                gene_1_innovation = self.generation_innovations[(old_input_type, old_input_node, new_output_type_1, new_output_node)]
                genome.links.add((old_input_type, old_input_node, new_output_type_1, new_output_node))
            else:
                gene_1_innovation = self.new_innovation()
                self.generation_innovations[(old_input_type, old_input_node, new_output_type_1, new_output_node)] = gene_1_innovation
                genome.links.add((old_input_type, old_input_node, new_output_type_1, new_output_node))
            if (new_input_type_2, new_input_node, old_output_type, old_output_node) in self.generation_innovations:
                gene_2_innovation = self.generation_innovations[(new_input_type_2, new_input_node, old_output_type, old_output_node)]
                genome.links.add((new_input_type_2, new_input_node, old_output_type, old_output_node))
            else:
                gene_2_innovation = self.new_innovation()
                self.generation_innovations[(new_input_type_2, new_input_node, old_output_type, old_output_node)] = gene_2_innovation
                genome.links.add((new_input_type_2, new_input_node, old_output_type, old_output_node))


            weight = old_gene.weight
            bias = old_gene.bias

            gene_1 = Gene(old_input_node, old_input_type, new_output_node, new_output_type_1, weight, bias,
                 enabled=True, innovation=gene_1_innovation, recurrent=temp_recurrent)
            gene_2 = Gene(new_input_node, new_input_type_2, old_output_node, old_output_type, weight, bias,
                 enabled=True, innovation=gene_2_innovation, recurrent=temp_recurrent)

            genome.add_gene(gene_1)
            genome.add_gene(gene_2)

    def mutate_weights(self, genome):
        ## Should it be perturbed or randomly set?
        temp_perturb_val = self.generator.random()
        ## Check if we are updating or setting weight and bias values.
        if temp_perturb_val < self.weight_perturbation_chance:
            ## We should perturb.
            for key, gene in enumerate(genome.genes):
                ## For updating the weight.
                temp_perturb_val = (self.generator.random() - 0.5) * 2.0 * self.max_weight_perturbation
                ## Add the perturbation to the gene's weight.
                gene.weight += temp_perturb_val

                ## For updating the bias.
                temp_perturb_val = (self.generator.random() - 0.5) * 2.0 * self.max_bias_perturbation
                ## Add the perturbation to the gene's bias.
                gene.bias += temp_perturb_val
        else:
            ## We should randomly set weights.
            for key, gene in enumerate(genome.genes):
                ## For updating the weight.
                temp_perturb_val = (self.generator.random() - 0.5) * 2.0 * self.max_weight_replaced
                ## Set the gene's weight to the random value.
                gene.weight = temp_perturb_val

                ## For updating the bias.
                temp_perturb_val = (self.generator.random() - 0.5) * 2.0 * self.max_bias_replaced
                ## Set the gene's bias to the random value.
                gene.bias = temp_perturb_val

    def mutate_genome(self, genome, genome_number, species_number):
        #print("Mutate test.")
        active_genome = copy.deepcopy(genome)
        ## Should this genome be copied unchanged?

        ## Should this genome breed?
        #print("test2")
        temp_breed_val = self.generator.random()
        if temp_breed_val > self.no_crossover_chance:
            ## It should breed
            ## Should it mate with a different species?
            temp_interspecies_val = self.generator.random()
            if temp_interspecies_val < self.inter_species_chance and len(self.species_list) > 1:
                ## It should be an interspecies breeding.
                ## Find a genome outside of this genome's species.
                temp_set = set(range(0, len(self.species_list)))
                temp_set.remove(species_number)
                temp_list = list(temp_set)
                species_chosen = self.generator.choice(temp_list)
                #print("spcecies chosen",species_chosen)
                temp_length = len(self.species_list[species_chosen].genomes)
                if len(self.species_list[species_chosen].genomes) > 1:
                    genome_chosen = self.generator.sample((0, temp_length - 1), 1)[0]
                else:
                    genome_chosen = 0
                #print("genome chosen", genome_chosen)
                #print("species size", len(self.species_list[species_chosen].genomes))
                active_genome = self.breed_genomes(genome, self.species_list[species_chosen].genomes[genome_chosen])
            else:
                ## It should be an intraspecies breeding.
                ## Find a genome inside this genome's species that
                ## is not this genome to breed with
                #print("Just before:",len(self.species_list[species_number].genomes))
                temp_set = set(range(0, len(self.species_list[species_number].genomes)))
                #print(temp_set)
                #for i in self.species_list[species_number].genomes:
                #    print("Genome number: {}".format(genome_number))
                temp_set.remove(genome_number)
                temp_list = list(temp_set)
                if len(self.species_list[species_number].genomes) > 1:
                    genome_chosen = self.generator.choice(temp_list)
                    active_genome = self.breed_genomes(genome, self.species_list[species_number].genomes[genome_chosen])


        ## Should the genome's weights be manipulated?
        temp_weight_val = self.generator.random()
        if temp_weight_val < self.weight_manipulation_chance:
            ## The weights should be manipulated.
            self.mutate_weights(active_genome)


        ## Should a new link be created or deleted?
        temp_link_val = self.generator.random()
        #print("test")

        if temp_link_val < self.link_deletion_chance:
            self.delete_link(active_genome)


        temp_link_val = self.generator.random()

        if temp_link_val < self.link_generation_chance:
            #print("Generate link.")
            ## Generate a new link.
            self.mutate_link(active_genome)

        ## Should a new node be created?
        temp_node_val = self.generator.random()
        if temp_node_val < self.node_generation_chance:
            #print("Generate node.")
            ## Generate a new node.
            self.mutate_node(active_genome)

        ## TODO: This is the function we would use to switch between
        ## growth and culling phases.

        return active_genome

    def process_new_generation(self):
        ## Clear the generation_innovations dictionary.
        self.generation_innovations = {}

        keep_species_list = []
        total_adjusted_fitness = 0.0

        ## Step through each species and decide which ones to keep.
        for key, species in enumerate(self.species_list):
            if species.stagnant_time < self.stagnant_time:
                species.remove_weak_genomes()
                species.calculate_adjusted_fitness()
                keep_species_list.append(species)

        keep_species_list.sort(key=lambda data: data.adjusted_fitness_sum, reverse=True)
        ## Cull down to the best
        if len(keep_species_list) >= self.max_num_species:
            keep_species_list = keep_species_list[:self.max_num_species]

        self.species_list = keep_species_list

        for species in self.species_list:
            total_adjusted_fitness += species.adjusted_fitness_sum
        #print(total_adjusted_fitness)
        #print(len(self.species_list))
        #input("total_adjusted_fitness")

        num_genomes_created = 0
        new_genomes = []
        new_species_list = []

        for species_key, species in enumerate(self.species_list):
            percent_allocated = float(species.adjusted_fitness_sum) / float(total_adjusted_fitness)
            genomes_allocated = math.floor(percent_allocated * self.population_size)
            #print(percent_allocated, genomes_allocated)
            #input("percent and genomes allocated")
            if genomes_allocated > 0:
                ## Actually produce some offspring.
                if genomes_allocated > 1:
                    ## Make sure to keep the best genome.
                    genomes_allocated -= 1
                    temp_genome = species.genomes[0]
                    ## TODO: Use this to add the old species to the new set.
                    new_species_list.append(Species(temp_genome, species.species_number))
                    ##new_genomes.append(copy.deepcopy(temp_genome))
                    num_genomes_created += 1
                for _ in range(genomes_allocated):
                    if len(species.genomes) > 1:
                        genome_key = self.generator.randint(0, len(species.genomes) - 1)
                        old_genome = species.genomes[genome_key]

                        new_genome = self.mutate_genome(old_genome, genome_key, species_key)
                        num_genomes_created += 1
                        new_genomes.append(new_genome)
        #print("New species list size: ", len(new_species_list))
        #input("...")

        if num_genomes_created < self.population_size:
            for _ in range(self.population_size - num_genomes_created):
                if len(self.species_list) == 1:
                    species_key = 0
                else:
                    species_key = self.generator.randint(0, len(self.species_list) - 1)
                if len(self.species_list[species_key].genomes) == 1:
                    genome_key = 0
                else:
                    genome_key = self.generator.randint(0, len(self.species_list[species_key].genomes) - 1)
                old_genome = self.species_list[species_key].genomes[genome_key]
                new_genome = self.mutate_genome(old_genome, genome_key, species_key)
                new_genomes.append(new_genome)

        self.species_list = new_species_list

        for genome in new_genomes:
            err = self.add_genome_to_species(genome)
            if err is not None:
                return err
        #print("New spcies list size: ", len(self.species_list))
        #input("...")

    def process_new_generation2(self):
        ## Returns error.
        ## Clear the generation_innovations dictionary.
        self.generation_innovations = {}
        ## Sort the species by their species number.
        ## For each species.
        species_to_keep = set()
        num_genomes_created = 0
        total_adjusted_fitness = 0
        keep_species_adjusted_sum = 0
        for key, species in enumerate(self.species_list):
            ##   Calculate the adjusted fitness.
            species.calculate_adjusted_fitness()
            #print(species.adjusted_fitness_sum)
            total_adjusted_fitness += species.adjusted_fitness_sum
            ## Check if the species has passed the stagnant limit.
        self.species_list.sort(key=lambda data: data.adjusted_fitness_sum, reverse=True)
        for key, species in enumerate(self.species_list):
            #if species.stagnant_time < self.stagnant_time and species.adjusted_fitness_sum > self.highest_fitness / 2 and len(species_to_keep) < self.max_num_species:
            if species.stagnant_time < self.stagnant_time and len(species_to_keep) < self.max_num_species:
                species_to_keep.add(key)
                keep_species_adjusted_sum += species.adjusted_fitness_sum
        new_species = []
        #self.species_list
        culled_species_data = []

        for key, species in enumerate(self.species_list):
            if key in species_to_keep:
                temp_species = Species(
                        copy.deepcopy(species.genomes[species.current_max_genome]),
                        species.species_number)
                new_species.append(temp_species)
                num_genomes_created += 1
                #print("Before cull: {}.".format(len(species.genomes)))
                genome_fitness_data = species.remove_weak_genomes()
                culled_species_data.append(genome_fitness_data)
                #print("After cull: {}.".format(len(species.genomes)))

        new_genomes = []
        for species_key, species in enumerate(self.species_list):
            if species_key in species_to_keep:
                #print("adjusted_fitness_sum", species.adjusted_fitness_sum)
                #print("total_adjusted_fitness", total_adjusted_fitness)
                percent_allocated = float(species.adjusted_fitness_sum) / float(keep_species_adjusted_sum)
                genomes_allocated = int(percent_allocated * self.population_size)
                #print("species fitness", species.adjusted_fitness_sum)
                #print("keep_species_adjusted_sum", keep_species_adjusted_sum)
                #print("test", percent_allocated)
                #print("test4", genomes_allocated)
                #print("Species to keep: ", species_to_keep)
                #print("num genomes added: ", len(new_genomes))
                #input("press button")
                ## Generate the new genomes for this species.
                for i in range(genomes_allocated):
                    try:
                        if len(culled_species_data[species_key]) > 1:
                            ## Select a random genome to mutate.
                            genome_key = self.generator.randint(0, len(culled_species_data[species_key]) - 1)
                            #print(len(species.genomes))
                            #print("old",genome_key)
                            #print("old2", len(culled_species_data[species_key]))
                            ##genome = species.genomes[genome_key]
                            genome = species.genomes[culled_species_data[species_key][genome_key]['key']]
                            #print("test3")
                            temp_genome = self.mutate_genome(genome, genome_key, species_key)
                            new_genomes.append(temp_genome)
                            ##err = self.add_genome_to_species(self.mutate_genome(genome,
                            ##        genome_key, species_key))
                            ##if err is not None:
                            ##    return err
                            num_genomes_created += 1
                        else:
                            genome_key = 0
                            genome = species.genomes[0]
                            temp_genome = self.mutate_genome(genome, genome_key, species_key)
                            new_genomes.append(temp_genome)
                            num_genomes_created += 1
                    except:
                        pass

        ## Create random genomes to replace the ones lost.

        num_genomes_left = self.population_size - num_genomes_created
        #print("num_genomes_left", num_genomes_left)

        for i in range(num_genomes_left):
            ## Add random genome to species.
            ## Pick a random species.
            random_species_key = self.generator.randint(0, len(self.species_list) - 1)
            random_genome_key = self.generator.randint(0, len(self.species_list[random_species_key].genomes)-1)
            random_genome = self.mutate_genome(copy.deepcopy(self.species_list[random_species_key].genomes[random_genome_key]), random_genome_key, random_species_key)
            #random_genome = self.starter_genome()
            new_genomes.append(random_genome)
            ##err = self.add_genome_to_species(random_genome)
            ##if err is not None:
            ##    return err
        self.species_list = new_species
        for genome in new_genomes:
            err = self.add_genome_to_species(genome)
            if err is not None:
                return err

        total_num_genomes = 0
        for species in self.species_list:
            total_num_genomes += len(species.genomes)
        #print("Total number of genomes: {}\nNumber of species: {}".format(total_num_genomes, len(self.species_list)))


        #self.current_generation += 1
        #self.generation_processed = False
        return None

    def process_data(self, input_data):
        ## Take data in.

        ## Process the data.
        #print("Generation: {}, Species: {}, Genome:{}, Frame:{}".format(self.current_generation, self.current_species, self.current_genome, self.current_frame))
        #for gene in self.species_list[self.current_species].genomes[self.current_genome].genes:
        #    print(gene.printable())
        #print("species list length: {}, genome length: {}".format(len(self.species_list), 0))
        ## If we are still going through the old path, just get the data from the old genome.
        if len(self.previous_path) == 0:
            ## Otherwise continue learning in this population.
            output_data = self.species_list[self.current_species].genomes[self.current_genome].run_network(input_data)
            temp_genes = self.species_list[self.current_species].genomes[self.current_genome].genes
        else:
            #print(self.previous_path)
            output_data = self.previous_path[0][0].run_network(input_data)
        #if len(temp_genes) > self.max_num_genes:
        #    print(len(temp_genes))
        #    for gene in temp_genes:
        #        print("Gene in_node:{}, in_type:{}, out_node:{}, out_type:{}, weight:{}, bias:{}".format(gene.in_node, gene.input_type, gene.out_node, gene.output_type, gene.weight, gene.bias))

        #    #input("Press a key to continue...")
        #    self.max_num_genes = len(temp_genes)

        ## Return the output of the current run.
        #print('output_data', output_data)
        return output_data

    def update_position(self):
        ## Check if it is time to create a new generation.
        ## Also, track current location in the population.
        self.current_frame += 1
        #print("genome processed: {}".format(self.genome_processed))
        if self.genome_processed: ## 4
            self.genome_processed = False
            self.current_genome += 1
            self.current_frame = 0

            if self.species_processed:
                self.species_processed = False
                self.current_species += 1
                self.current_genome = 0

                if self.generation_processed:
                    self.generation_processed = False
                    self.current_generation += 1
                    self.current_species = 0
        '''
            if self.current_species == len(self.species_list) - 1:
                if self.current_genome == len(self.species_list[self.current_species].genomes):
                    #self.species_list[self.current_species].genomes[self.current_genome].fitness = last_reward
                    ## We need to create a new generation.
                    self.current_species = 0
                    self.current_genome = 0
                    ## Create new generation.
                    ##self.process_new_generation()
                    self.generation_processed = True

            else:
                #self.species_list[self.current_species].genomes[self.current_genome].fitness = last_reward
                if self.current_genome == len(self.species_list[self.current_species].genomes):
                    self.current_species += 1
                    self.current_genome = 0
            self.current_frame = 0
        '''

    def check_location(self):
        if len(self.previous_path) == 0:
            if self.current_frame >= self.frames_to_update + self.extra_frames_earned - 1:
                self.extra_frames_earned = 0
                self.genome_processed = True
                #print("Genome processed!")

                if self.current_genome == len(self.species_list[self.current_species].genomes) - 1:
                    self.species_processed = True
                    if self.current_species == len(self.species_list) - 1:
                        #print("test45")
                        self.generation_processed = True
        else:
            ## Check if we need to move on from this part of the path.
            x_val = self.previous_path[0][1][0]
            y_val = self.previous_path[0][1][1]
            x_high_val = x_val + abs(x_val / 25.0)
            x_low_val = x_val - abs(x_val / 25.0)
            y_high_val = y_val + abs(y_val / 25.0)
            y_low_val = y_val - abs(y_val / 25.0)
            ## Check if our current position is within four percent of the current goal.
            if current_position[0] < x_high_val and current_position[0] > x_low_val:
                if current_position [1] < y_high_val and current_position > y_low_val:
                    self.previous_path.pop(0)



    def apply_reward(self, reward):
        self.species_list[self.current_species].genomes[self.current_genome].fitness = reward
        if reward > self.highest_fitness:
            self.highest_fitness = reward
            self.best_genome = copy.deepcopy(self.species_list[self.current_species].genomes[self.current_genome])

    def print_current(self):
        current_genome = self.species_list[self.current_species].genomes[self.current_genome]
        #for gene in current_genome.genes:
        #    print(gene.printable())

if __name__ == '__main__':
    ## Generate XOR data.
    inputs = []
    for i in range(2):
        for j in range(2):
            inputs.append([i, j])

    test_population = Population(2, 1, None, len(inputs))
    test_population.setup()

    fitness = 0
    for i in range(80000):
        temp_index = i % len(inputs)
        #if temp_index == 0:
        #    print("Generation: {}, Species: {}, Genome:{}, Frame:{}".format(test_population.current_generation, test_population.current_species, test_population.current_genome, test_population.current_frame))

        ## First run our current position.
        #print("Species size:{}".format(len(test_population.species_list[test_population.current_species].genomes)))
        #print("Number of Species:{}".format(len(test_population.species_list)))
        #print("Generation processed:{}".format(test_population.generation_processed))
        #print("test42")
        temp_output = test_population.process_data(inputs[temp_index])[0]
        actual_output = inputs[temp_index][0] ^ inputs[temp_index][1]
        #error = - ((temp_output - actual_output)*(temp_output - actual_output))
        #fitness += ( 1 - math.pow(actual_output - temp_output, 2))
        fitness += (1 - abs(actual_output - temp_output))
        #print("Prediction: {}. Actual: {}\nFitness: {}".format(temp_output, actual_output, fitness))
        #test_population.apply_reward(fitness

        ## Next check if we are at the end of one of the groups.
        #print("test43")
        test_population.check_location()

        if test_population.genome_processed:
            #test_population.apply_reward(fitness)

            temp_fitness = 0.0
            for l in range(4):
                temp_output = test_population.process_data(inputs[l])[0]
                actual_output = inputs[l][0] ^ inputs[l][1]
                #temp_fitness += (1 - math.pow(actual_output - temp_output, 2))
                temp_fitness += (1 - abs(actual_output - temp_output))
                #if temp_fitness > 3.0:
                #    #print("Prediction: {}. Actual: {}\nFitness: {}".format(temp_output, actual_output, temp_fitness))
                #    #test_population.print_current()
                #    #input("...")

            #if temp_fitness > 2.0:
            #    print("temp_fitness: {}".format(temp_fitness))
            #    print("fitness: {}".format(fitness))
            test_population.apply_reward(temp_fitness)
            fitness = 0


        if test_population.generation_processed:
            test_population.process_new_generation()
            print("highest fitness",test_population.highest_fitness)

        ## Move to the next position.
        #print("test44")
        test_population.update_position()
    print(test_population.highest_fitness)
    print(test_population.max_num_genes)
