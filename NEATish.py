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
        ## That doesn't fit with any of the other species. This means
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
        ## Find the adjusted fitness of each genome then sum them.
        for genome in genomes:
            genome.calculate_adjusted_fitness(self.current_size)
            adjusted_sum += genome.adjusted_fitness
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

    def add_gene(self, gene):
        ## Add a new gene to the genome.
        self.genes.append(gene)

    def sort_genes(self):
        ## Sort genes within the genome for crossover analysis.
        self.genes = sorted(self.genes, key=lambda gene: gene.innovation)

    def calculate_adjusted_fitness(self, species_size):
        ## The adjusted fitness is the fitness of the genome divided
        ## by the number of members in the species it belongs to.
        self.adjusted_fitness = self.fitness / species_size

    def create_network(self):
        self.neuron_hidden_nodes = [Node()] * self.num_hidden_nodes
        self.neuron_output_nodes = [Node()] * self.num_outputs
        for gene in self.genes:
            if gene.output_type == "HIDDEN":
                self.neuron_hidden_nodes[gene.out_node].input_connections.append(gene)
            if gene.output_type == "OUTPUT":
                self.neuron_output_nodes[gene.out_node].input_connections.append(gene)

    def run_network(self, inputs):
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
                        temp_in_value = self.neuron_hidden_nodes[temp_in_node].value
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
                        temp_in_value = self.neuron_hidden_nodes[temp_in_node].value
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
    def __init__(self, num_inputs, num_outputs):
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
        ## number. Each gene will be given as {(input, output): innovation}
        self.generation_innovations = {}
        ## TODO: I think this is right? the basic genome should have
        ## num_inputs * num_outputs genes so new innovations should
        ## start past there.
        self.global_innovation = num_inputs * num_outputs

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

    def starter_genome(self):
        ## Returns a new basic genome.
        ## Make a new genome with the right number of inputs and outputs.
        new_genome = Genome(self.num_inputs, self.num_outputs)
        ## Each basic genome will have the same innovations to fully
        ## connect them.
        temp_innovations = 0
        ## Fully connect the network.
        for i in range(self.num_inputs):
            for j in range(self.num_outputs):
                temp_weight = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                temp_bias = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
                temp_gene = Gene(i, "INPUT", j, "OUTPUT", temp_weight,
                                 temp_bias, enabled=True,
                                 temp_innovations)
                new_genome.genes.append(temp_gene)
                temp_innovations += 1

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
        return new_genome


    def mutate_link(self, genome):
        ## TODO: Finish documenting.
        ## Connect two random nodes to each other.
        ## Check if these two are connected already.
        ## Find the number of hidden nodes.
        temp_counter = 0
        for gene in genome.genes:
            if gene.output_type == "HIDDEN":
                temp_counter += 1
        random_index = self.generator.randrange(0, genome.num_inputs + genome.num_outputs + temp_counter)
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

        random_index = self.generator.randrange(0, genome.num_inputs + genome.num_outputs + temp_counter)
        if random_index < genome.num_inputs:
            ## Use an input as the input node.
            out_node = random_index
            output_type = "INPUT"
            pass
        elif random_index < genome.num_inputs + genome.num_outputs:
            ## Use an output as the input node.
            output_node = random_index - genome.num_inputs
            output_type = "OUTPUT"
        else:
            ## Use a hidden node as the input node.
            output_node = random_index - genome.num_inputs - genome.num_outputs
            output_type = "HIDDEN"

        temp_innovation = self.new_innovation
        if input_type == output_type and in_node == out_node:
            temp_reccurent = True
        else:
            temp_reccurent = False
        temp_weight = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
        temp_bias = (self.generator.random() - 1.0) * 2.0 * self.max_weight_replaced
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
        ## TODO: Fill this in.
        genome_length = len(genome.genes)

        ## Our current gene.
        random_index = self.generator.randrange(0, genome_length)
        ## Dissable this gene.
        genome.genes[random_index].enabled = False
        ## Create two new genes using the input and output of the current
        ## gene.
        ## This deals with the current input.

        pass

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

    def mutate_genome(self, genome, genome_number):
        ## Should this genome be copied unchanged?

        ## Should this genome breed?
        temp_breed_val = self.generator.random()
        if temp_breed_val > self.no_crossover_chance:
            ## It should breed
            ## Should it mate with a different species?
            temp_interspecies_val = self.generator.random()
            if temp_interspecies_val < self.inter_species_chance:
                ## It should be an interspecies breeding.
                ## TODO: Find a genome outside of this genome's species.
                pass
            else:
                ## It should not be an intraspecies breeding.
                ## TODO: Find a genome inside this genome's species that
                ## is not this genome to breed with.
                pass

        ## Should the genome's weights be manipulated?
        temp_weight_val = self.generator.random()
        if temp_weight_val < self.weight_manipulation_chance:
            ## The weights should be manipulated.
            ## TODO: Call mutate weights.
            pass


        ## Should a new link be created?
        temp_link_val = self.generator.random()
        if temp_link_val < self.link_generation_chance:
            ## Generate a new link.
            pass

        ## Should a new node be created?
        temp_node_val = self.generator.random()
        if temp_node_val < self.node_generation_chance:
            ## Generate a new node.
            pass

        ## TODO: This is the function we would use to switch between
        ## growth and culling phases.

    def process_generation(self):
        ## TODO: Actually fill this out.
        ## Sort the species by their species number.
        ## For each species:
        ##   Sort the genomes by genome_number
        ##   For each genome in the species
        ##     Build the network for the current genome
        ##     Run simulator for one run.
        ##     **Remember to do something about novelty search**
        ##     Run fitness information.
        ## For each species.
        ##   Calculate the adjusted fitness.
        ## Resample based on adjusted fitness.
        ##   For i in range of number of genomes to reproduce:
        ##     randomly select a genome in the species and reproduce it
        pass
