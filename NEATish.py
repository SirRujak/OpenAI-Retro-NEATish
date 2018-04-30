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

class Activations:
    ## TODO: Documentation.
    def __init__(self, default="SIGMOID"):
        self.default = default

    def base_sigmoid(self, input_val):
        ## This is just calculating a sigmoid:
        ## y = 1/(1 + e^(-ax))
        power = -4.9 * input_val
        bottom = 1.0 + exp(power)
        result = 1.0 / bottom
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
        self.genomes = [initial_genomes]
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
        for key, genome in enumerate(genomes):
            genome.calculate_adjusted_fitness(self.current_size)
            if genome.adjusted_fitness > max_adjusted_fitness:
                max_adjusted_fitness = genome.adjusted_fitness
                max_key = key
            adjusted_sum += genome.adjusted_fitness
        self.current_max_genome = max_key
        self.adjusted_fitness_sum = adjusted_sum

        ## Check if the species made progress this generation.
        if self.adjusted_fitness_sum > self.maximum_adjusted_fitness:
            self.stagnant_time = 0
        else:
            self.stagnant_time += 1

    def remove_weak_genomes(self, all_but_one=False):
        self.genomes.sort(key=lambda genome: genome.adjusted_fitness, reverse=True)
        if not all_but_one:
            num_to_keep = math.ceil(len(self.genomes)//2)
        else:
            num_to_keep = 1

        self.genomes = self.genomes[:num_to_keep]



class Genome:
    ## TODO: Documentation.
    def __init__(self, num_inputs=0, num_outputs=0, genes=[]
                 neuron_input_nodes=[], neuron_output_nodes=[],
                 neuron_hidden_nodes=[], fitness=0.0 species=None,
                 genome_number=0):
        ## TODO: Add some documentation here.
        self.activations = Activations()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genes = genes
        if len(self.neuron_input_nodes) != 0:
            self.neuron_input_nodes = neuron_input_nodes
        else:
            for i in range(num_inputs):
                self.neuron_input_nodes.append(Node())
        if len(self.neuron_output_nodes) != 0:
            self.neuron_output_nodes = neuron_output_nodes
        else:
            for i in range(num_outputs):
                self.neuron_output_nodes.append(Node())

        self.neuron_hidden_nodes = neuron_hidden_nodes
        self.num_hidden_nodes = len(neuron_hidden_nodes)


        self.fitness = fitness
        self.adjusted_fitness = fitness
        self.genome_number = genome_number


        self.active_gene_set = set()
        self.active_gene_dict = {}

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
        self.adjusted_fitness = self.fitness / species_size

    def create_network(self):
        hidden_node_set = set()
        current_hidden_node_counter = 0
        self.hidden_node_dict = {}
        for gene in self.genes:
            if gene.output_type == "HIDDEN":
                if gene.out_node not in hidden_node_set:
                    self.hidden_node_dict[gene.out_node] = current_hidden_node_counter
                    current_hidden_node_counter += 1
                hidden_node_set.add(gene.out_node)

        for key, temp_node_number in enumerate(list(hidden_node_set)):
            self.hidden_node_dict[temp_node_number] = key


        self.num_hidden_nodes = len(hidden_node_set)

        self.neuron_hidden_nodes = [Node()] * self.num_hidden_nodes
        self.neuron_output_nodes = [Node()] * self.num_outputs
        for gene in self.genes:
            if gene.output_type == "HIDDEN":
                self.neuron_hidden_nodes[self.hidden_node_dict[gene.out_node]].input_connections.append(gene)
            if gene.output_type == "OUTPUT":
                self.neuron_output_nodes[gene.out_node].input_connections.append(gene)

    def run_network(self, inputs):
        if len(self.neuron_hidden_nodes) == 0:
            self.sort_genes()
            self.create_network()
        for i in range(self.num_inputs):
            self.neuron_input_nodes[i].value = inputs[i]

        for node in self.neuron_hidden_nodes:
            temp_sum = 0.0
            for connection in node.input_connections:
                if connection.enabled:
                    temp_in_node = connection.in_node
                    if connection.input_type == "INPUT":
                        temp_in_value = self.neuron_input_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias
                    elif connection.input_type == "HIDDEN":
                        temp_in_value = self.neuron_hidden_nodes[self.hidden_node_dict[temp_in_node]].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias
                    elif connection.input_type == "OUTPUT":
                        temp_in_value = self.neuron_output_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias

            node.value = self.activations.base_sigmoid(temp_sum)

        output = [0.0] * self.num_outputs
        for key, node in enumerate(self.output_nodes):
            temp_sum = 0.0
            for connection in node.input_connections:
                if connection.enabled:
                    temp_in_node = connection.in_node
                    if connection.input_type == "INPUT":
                        temp_in_value = self.neuron_input_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias
                    elif connection.input_type == "HIDDEN":
                        temp_in_value = self.neuron_hidden_nodes[self.hidden_node_dict[temp_in_node]].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias
                    elif connection.input_type == "OUTPUT":
                        temp_in_value = self.neuron_output_nodes[temp_in_node].value
                        temp_weight = connection.weight
                        temp_bias = connection.bias
                        temp_sum += temp_in_value * temp_weight + temp_bias

            node.value = self.activations.base_sigmoid(temp_sum)
            output[key] = node.value
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
        return "into:{}, output:{}, weight:{}, bias:{}, enabled:{}".format(
            self.in_node, self.out_node, self.weight, self.bias, self.enabled
        )

class Node:
    ## TODO: Documentation.
    def __init__(self, input_connections=set(), output_connections=set(),
                 hidden_connections=set(), value=0.0):
        self.input_connections = input_connections
        self.output_connections = output_connections
        self.hidden_connections = hidden_connections
        self.value = value

class Population:
    ## TODO: Documentation.
    def __init__(self, num_inputs, num_outputs, frames_to_update=100,
                 population_size=300):
        ## Generator with seed. If a seed is given it should be
        ## deterministic.
        if seed:
            self.seed = seed
            self.generator = random.Random()
            self.generator.seed(seed)
        else:
            self.seed = None
            self.generator = random.Random()
        ## Constants used to separate genomes into species.
        ## Constant for excess genes.
        self.c1 = 1.0
        ## Constant for disjoint genes.
        self.c2 = 1.0
        ## By increasing this the system accounts for more variations
        ## in gene weights
        self.c3 = 0.4 ## 3.0 for large populations

        ## Constant for acceptable degree of differentiation between
        ## species
        self.del_t = 3.0 ## 4.0 for large populations

        ## If species have been stagnant for this amount of time
        ## that species is no longer allowed to reproduce. Stagnant
        ## is defined by no increase in fitness.
        self.stagnant_time = 15

        ## Number of members of each species to copy champion with
        ## no changes.
        self.champ_species_size = 5

        ## Chance a genome will have weights manipulated.
        self.weight_manipulation_chance = 0.8

        ## This is used for deciding perturbation or full random changes.
        ## This is only used if the weight manipulation check passes.
        ## 90% perturbation chance, 10% random assignment chance.
        self.weight_perturbation_chance = 0.9

        ## If the gene is disabled in either parent there is this chance
        ## that the child will also have the gene disabled.
        self.disable_chance = 0.75

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

        ## TODO: Should we add in the ability to do contraction
        ## manipulations? We would need something for keeping track
        ## of how long we have been on each phase of generation and
        ## culling, how well we have been progressing recently,
        ## and which stage we are on.

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

        ## The maximum amount a mutation can perturb a weight.
        self.max_weight_perturbation = 1.0
        ## The maximum amount a mutation can perturb a bias.
        self.max_bias_perturbation = 1.0

        ## The maximum value of a replaced weight.
        self.max_weight_replaced = 1.0
        ## The maximum value of a replaced bias.
        self.max_bias_replaced = 1.0

        ## Number of inputs to the network.
        self.num_inputs = num_inputs

        ## Number of outputs for the network.
        self.num_outputs = num_outputs

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

        ## Max number of frames to use per network run.
        self.frames_to_update = frames_to_update

        ## Value for checking if two genomes should be of the same species.
        self.compatability_cutoff = 1

        self.species_counter = 0

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
        ## Calculate compatability between two genes and return it.
        compatability_distance = ((self.c1 * excess + self.c2 * disjoint) \
            / N) + self.c3 * W
        return (compatibility_distance, None)

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
            if current_position_1 == genome_len_1 and current_position_2 == genome_len_2:
                ## If both gene postions are the final positions we are
                ## done.
                not_done = False
                ## In this case if the current genemoe innovation is not
                ## the same it means we have one more excess gene.
                if genome_1.genes[current_position_1].innovation != genome_2.genes[current_position_2].innovation:
                    excess += 1
            elif (genome_1.genes[current_position_1].innovation ==
            genome2.genes[current_position_2].innovation):
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
            genome2.genes[current_position_2].innovation):
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
            genome2.genes[current_position_2].innovation)
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

        w = float(weight_differences_sum) / float(num_matching_genes)
        return (excess, disjoint, n, w)

    def add_genome_to_species(self, genome):
        for key, species in self.species_list:
            ## TODO: Make sure to delete species if they don't have any genomes.
            for species in self.species_list:
                compatability, err = self.calculate_compatibility(species.representative, genome)
                if err is not None:
                    return err
                else:
                    if compatability > self.compatability_cutoff:
                        species.genomes.append(genome)
                        return None
            ## If we were unable to find one. Create a new species and add it.
            self.species_list.append(Species(genome, self.species_counter))
            self.species_counter += 1
            return None

    def generate_initial_population(self):
        ## TODO: Do things!
        for i in range(self.population_size):
            self.add_genome_to_species(self.starter_genome)

    def starter_genome(self, num_input_connections=5, num_output_connections=5):
        ## Returns a new basic genome.
        ## Make a new genome with the right number of inputs and outputs.
        new_genome = Genome(self.num_inputs, self.num_outputs)

        '''
        ## Each basic genome will have the same innovations to fully
        ## connect them.
        temp_innovations = 0

        ## Fully connect the network.
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                ## Random weight and bias information.
                temp_weight = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                temp_bias = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
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
        output_values = self.generator.sample(range(0, self.num_outputs), num_output_connections)

        for i in range(num_input_connections):
            for j in range(num_output_connections):
                ## Dealing with the innovation.
                if ("INPUT", i, "OUTPUT", j) in self.generation_innovations:
                    temp_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
                else:
                    temp_innovation = self.new_innovation()
                    self.generation_innovations[(input_type, input_node, output_type, output_node)] = temp_innovation

                ## Random weight and bias information.
                temp_weight = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                temp_bias = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                temp_gene = Gene(i, "INPUT", j, "OUTPUT", temp_weight,
                                 temp_bias, enabled=True,
                                 temp_innovation)
                ## This will be used in node mutation later.
                temp_len = len(new_genome.genes)
                new_genome.active_gene_dict[temp_innovation] = temp_len
                new_genome.active_gene_set.add(temp_innovation)
                new_genome.genes.append(temp_gene)
                temp_innovations += 1

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
        for key, gene in enumerate(genome_1):
            genome_1_dict[gene.innovation] = key
        for key, gene in enumerate(genome_2):
            genome_2_dict[gene.innovation] = key

        ## Genome to return.
        new_genome = Genome()
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
                        new_gene = copy.deepcopy(gene))
                    else:
                        ## Take from the second.
                        new_gene = copy.deepcopy(genome_2.genes[genome_2_dict[gene.innovation]])

                ## Copy the disjoint and excess genes from genome_1.
                else:
                    new_gene = copy.deepcopy(gene)

                ## If genome is dissabled in either parent there is a
                ## chance of reviving it in the child.
                if not new_gene.enabled:
                    random_guess = self.generator.random()
                    if random_guess > self.disabled_chance:
                        new_gene.enabled = True
                new_genome.append(new_gene)

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
                        new_gene = copy.deep_copy(gene)
                    else:
                        ## Take from the first.
                        new_gene = copy.deepcopy(genome_1.genes[genome_1_dict[gene.innovation]])

                ## Copy the disjoint and excess genes from genome_2.
                else:
                    new_gene = copy.deepcopy(gene)

                ## If genome is dissabled in either parent there is a
                ## chance of reviving it in the child.
                if not new_gene.enabled:
                    random_guess = self.generator.random()
                    if random_guess > self.disabled_chance:
                        new_gene.enabled = True
                new_genome.append(new_gene)

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
                        new_gene = copy.deep_copy(gene)
                    else:
                        ## Take from the second.
                        new_gene = copy.deep_copy(genome_2.genes[genome_2_dict[gene.innovation]])
                    del genome_2_dict[gene.innovation]

                else:
                    ## Always keep it.
                    new_gene = copy.deep_copy(gene)
                ## Check if there is a new gene to add.
                if new_gene:
                    ## Check if it is disabled and should be enabled.
                    if not new_gene.enabled:
                        random_guess = self.generator.random()
                        if random_guess > self.disabled_chance:
                            new_gene.enabled = True
                    new_genome.append(new_gene)

            for gene in genome_2.genes:
                if gene.innovation in genome_2_dict:
                    ## It is a disjoint or excess gene in genome 2.
                    ## Add it to the new genome.
                    new_gene = copy.deepcopy(gene)
                    ## Check if it is disabled and should be enabled.
                    if not new_gene.enabled:
                        random_guess = self.generator.random()
                        if random_guess > self.dissabled_chance:
                            new_gene.enabled = True
                    new_genome.append(new_gene)
        for key, gene in enumerate(new_genome.genes):
            if gene.enabled:
                new_genome.active_gene_dict[gene.innovation] = key
                new_genome.active_gene_set.add(gene.innovation)
        return new_genome


    def mutate_link(self, genome):
        ## TODO: Finish documenting.
        ## This does not add any new nodes into the network.
        ## Connect two random nodes to each other.
        ## Check if these two are connected already.
        ## Only output to hidden and output nodes.
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
        if (input_type, input_node, output_type, output_node) in self.generation_innovations:
            temp_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
        else:
            temp_innovation = self.new_innovation()
            self.generation_innovations[(input_type, input_node, output_type, output_node)] = temp_innovation
        if input_type == output_type and in_node == out_node:
            temp_reccurent = True
        else:
            temp_reccurent = False
        temp_weight = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
        temp_bias = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
        temp_len = len(genome.genes)
        genome.active_gene_dict[temp_innovation] = temp_len
        genome.active_gene_set.add(temp_innovation)
        genome.genes.append(Gene(input_node, input_type,
                                 output_node, output_type,
                                 weight, bias, enabled=True,
                                 innovation=temp_innovation,
                                 temp_reccurent))

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
            old_gene = genome.genes[genome.active_gene_dict[gene_innovation]]
            genome.active_gene_set.remove(gene_innovation)
            genome.active_gene_dict.remove(gene_innovation)
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

            if (input_type, input_node, output_type, output_node) in self.generation_innovations:
                gene_1_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
            else:
                gene_1_innovation = self.new_innovation()
                self.generation_innovations[(input_type, input_node, output_type, output_node)] = gene_1_innovation
            if (input_type, input_node, output_type, output_node) in self.generation_innovations:
                gene_2_innovation = self.generation_innovations[(input_type, input_node, output_type, output_node)]
            else:
                gene_2_innovation = self.new_innovation()
                self.generation_innovations[(input_type, input_node, output_type, output_node)] = gene_2_innovation

            new_output_type_1 = "HIDDEN"
            new_input_type_2 = "HIDDEN"

            new_output_node = len(genome.genes)
            new_input_node = len(genome.genes)

            weight = old_gene.weight
            bias = old_gene.bias

            gene_1 = Gene(old_input_node, old_input_type, new_output_node, new_output_type_1, weight, bias,
                 enabled=True, innovation=gene_1_innovation, temp_reccurent)
            gene_2 = Gene(new_input_node, new_input_type_2, old_output_node, old_output_type, weight, bias,
                 enabled=True, innovation=gene_2_innovation, temp_reccurent)

    def mutate_weights(self, genome):
        ## Should it be perturbed or randomly set?
        temp_perturb_val = self.generator.random()
        ## Check if we are updating or setting weight and bias values.
        if temp_perturb_val < self.weight_perturbation_chance
            ## We should perturb.
            for key, gene in enumerate(genome.genes):
                ## For updating the weight.
                temp_perturb_val = (self.generator.random() - 1.0) * 2.0 * self.max_weight_perturbation
                ## Add the perturbation to the gene's weight.
                gene.weight += temp_perturb_val

                ## For updating the bias.
                temp_perturb_val = (self.generator.random() - 1.0) * 2.0 * self.max_bias_perturbation
                ## Add the perturbation to the gene's bias.
                gene.bias += temp_perturb_val
        else:
            ## We should randomly set weights.
            for key, gene in enumerate(genome.genes):
                ## For updating the weight.
                temp_perturb_val = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                ## Set the gene's weight to the random value.
                gene.weight = temp_perturb_val

                ## For updating the bias.
                temp_perturb_val = (self.generator.random() - 1.0) * 2.0 * self.max_bias_replaced
                ## Set the gene's bias to the random value.
                gene.bias = temp_perturb_val

    def mutate_genome(self, genome, genome_number, species_number):
        active_genome = copy.deepcopy(genome)
        ## Should this genome be copied unchanged?

        ## Should this genome breed?
        temp_breed_val = self.generator.random()
        if temp_breed_val > self.no_crossover_chance:
            ## It should breed
            ## Should it mate with a different species?
            temp_interspecies_val = self.generator.random()
            if temp_interspecies_val < self.inter_species_chance:
                ## It should be an interspecies breeding.
                ## Find a genome outside of this genome's species.
                temp_set = set(range(0, len(self.species_list)))
                temp_set.remove(species_number)
                temp_list = list(temp_set)
                species_chosen = self.generator.choice(temp_list)
                temp_length = len(self.species_list[species_chosen].genomes)
                genome_chosen = self.generator.sample((0, temp_length), 1)
                active_genome = self.breed_genomes(genome, self.species_list[species_chosen].genomes[genome_chosen])
            else:
                ## It should be an intraspecies breeding.
                ## Find a genome inside this genome's species that
                ## is not this genome to breed with.
                temp_set = set(range(0, len(self.species_list[species_number])))
                temp_set.remove(genome_number)
                temp_list = list(temp_set)
                genome_chosen = self.generator.choice(temp_list)
                active_genome = self.breed_genomes(genome, self.species_list[speciesnumber].genomes[genome_chosen])


        ## Should the genome's weights be manipulated?
        temp_weight_val = self.generator.random()
        if temp_weight_val < self.weight_manipulation_chance:
            ## The weights should be manipulated.
            self.mutate_weights(active_genome)


        ## Should a new link be created?
        temp_link_val = self.generator.random()
        if temp_link_val < self.link_generation_chance:
            ## Generate a new link.
            self.mutate_link(active_genome)

        ## Should a new node be created?
        temp_node_val = self.generator.random()
        if temp_node_val < self.node_generation_chance:
            ## Generate a new node.
            self.mutate_node(active_genome)

        ## TODO: This is the function we would use to switch between
        ## growth and culling phases.

        return active_genome

    def process_new_generation(self):
        ## TODO: Actually fill this out.
        ## Clear the generation_innovations dictionary.
        self.generation_innovations = {}
        ## Sort the species by their species number.
        ## For each species.
        species_to_keep = set()
        num_genomes_created = 0
        total_adjusted_fitness = 0
        for key, species in enumerate(self.species_list):
            ##   Calculate the adjusted fitness.
            species.calculate_adjusted_fitness()
            total_adjusted_fitness += species.adjusted_fitness_sum
            species.remove_weak_genomes()
            ## Check if the species has passed the stagnant limit.
            if species.stagnant_time < self.stagnant_time:
                species_to_remove.add(key)
        old_species = self.species_list
        self.species_list = []

        for key, species in enumerate(old_species):
            if key not in species_to_remove:
                temp_species = Species(
                        copy.deepcopy(species.genomes[species.current_max_genome]),
                        species.species_number)
                self.species_list.append(temp_species)
                num_genomes_created += 1


        for species_key, species in enumerate(old_species):
            if species_key not in species_to_remove:
                percent_allocated = species.adjusted_fitness_sum // total_adjusted_fitness
                genomes_allocated = percent_allocated * self.population_size

                ## Generate the new genomes for this species.
                for i in range(genomes_allocated - 1):
                    ## Select a random genome to mutate.
                    genome_key = self.generator.randint(0, len(species.genomes))
                    genome = species.genomes[genome_key]
                    self.add_genome_to_species(self.mutate_genome(genome,
                            genome_key, species_key))
                    num_genomes_created += 1
                pass
        ## Create random genomes to replace the ones lost.

        num_genomes_left = self.population_size - num_genomes_created

        for i in range(num_genomes_left):
            ## Add random genome to species.
            random_genome = self.starter_genome()
            self.add_genome_to_species(random_genome)
        self.current_generation += 1

    def process_data(self, input_data, last_reward):
        ## Take data in.
        ## Check if it is time to create a new generation.
        ## Also, track current location in the population.
        if self.current_frame = self.frames_to_update:
            if self.current_species == len(self.species_list) - 1:
                if self.current_genome == len(self.species_list[self.current_species].genomes) - 1:
                    ## We need to create a new generation.
                    self.current_species = 0
                    self.current_genome = 0
                    ## Create new generation.
                    self.process_new_generation()
            else:
                ## Add last reward to the current genome.
                self.species_list[self.current_species].genomes[self.current_genome].fitness = last_reward
                if self.current_genome == len(self.species_list[self.current_species].genomes) - 1:
                    self.current_species += 1
                    self.current_genome = 0
                else:
                    self.current_genome += 1
            self.current_frame = 0

        ## Process the data.
        output_data = self.species_list[self.current_species].genomes[self.current_genome].run_network(input_data)
        ## Return the output of the current run.
        return output_data
