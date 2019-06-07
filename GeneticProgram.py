import numpy as np
from numpy.random import randint, uniform
from scipy.special import logsumexp
import copy
import pickle


def softmaxCrossEntropy(z, label):
    '''Multiclass error function.
    Softmax cross entropy is a combination of the softmax
    function, used to smooth a multiclass prediction vector
    into a valid probability distribution, and the generalized
    formula for cross-entropy entropy between the predicted and
    real distribution.

    Params:
    z:     Prediction vector (list of floats)
    label: Integer index of the true label
    '''
    return logsumexp(z) - z[label]

def sigmoid(z):
    '''Logistic function.

    Params:
    z: Single floating value input to the logistic function
    '''
    return 1.0 / (1.0 + np.exp(-z))

def logisticCrossEntropy(z, label):
    '''Binary classification error function.
    Logistic cross entropy is a combination of the logistic
    function, used to compress a binary class prediction value
    into a valid probability, and the cross-entropy error function
    for the binary classification case.

    Params:
    z:     Prediction value (single float)
    label: Integer value of the true label (0 or 1)
    '''
    y = sigmoid(z)
    t = float(label)
    return -1.0*t*np.log(y) - (1.0 - t)*np.log(1.0 - y)



class Program(object):
    '''Class implementing a canonical genetic program as described in
    the first CSCI 6506 sandbox, ie. consisting of ONLY the +, -, * and /
    operators and used ofr binary or multiclass classification.
    '''

    REGISTER_MODE              = 0
    INPUT_MODE                 = 1
    NUM_MODES                  = 2

    ADDITION_OP                = 0
    SUBTRACTION_OP             = 1
    MULTIPLICATION_OP          = 2
    DIVISION_OP                = 3
    NUM_OP_CODES               = 4

    OP_SYMBOLS                 = ['+', '-', '*', '/']

    MODE_INDEX                 = 0
    TARGET_INDEX               = 1
    OP_CODE_INDEX              = 2
    SOURCE_INDEX               = 3

    SAFE_DIVISION_RESULT       = 0.0

    NUM_INSTRUCTION_COMPONENTS = 4

    def __init__(self,
                 max_initial_instructions = 64,
                 num_registers            = 8,
                 num_inputs               = 4,
                 mutation_rate            = 0.1,
                 max_num_instructions     = 1024,
                 num_classes              = 3,
                 initialize_instructions  = True):

        # Initialize program parameters
        self._max_initial_instructions = max_initial_instructions
        self._num_registers            = num_registers
        self._num_inputs               = num_inputs
        self._mutation_rate            = mutation_rate
        self._max_num_instructions     = max_num_instructions
        self._num_classes              = num_classes


        # Pre-check on maximum source value range
        self._max_source_range         = max(self._num_inputs, self._num_registers)

        # Pre-calculate mod value depending on source access mode
        self._source_mod_value = [0, 0]
        self._source_mod_value[Program.INPUT_MODE]    = self._num_inputs
        self._source_mod_value[Program.REGISTER_MODE] = self._num_registers

        # Allocate space for registers
        self._registers                = np.zeros(self._num_registers)

        # Initialize random instructions. This is technically optional to speed
        # up re-production: when a child or children are produced, they will receive
        # (modified) copies of their parents' instructions and so don't need to
        # waste cycles initializing their own.
        self._instructions = []
        if initialize_instructions:
            n = randint(self._num_registers, self._max_initial_instructions + 1)
            for i in range(n):
                self._instructions.append(self.createRandomInstruction())


    @property
    def registers(self):
        return self._registers


    @property
    def instructions(self):
        return self._instructions


    @instructions.setter
    def instructions(self, new_instructions):
        self._instructions = new_instructions


    def printInstructions(self):
        '''Print out instructions in a readable format.'''
        for instruction in self.instructions:
            self.printInstruction(instruction)


    def printInstruction(self, instruction):
        mode, target_index, op_code, source_index = instruction

        source = None
        if mode == Program.REGISTER_MODE:
            source = 'R'
        else:
            source = 'IP'

        source_index %= self._source_mod_value[mode]

        instruction_string = 'R[' + \
                             str(target_index) + \
                             '] <- ' + \
                             'R[' + \
                             str(target_index) + \
                             '] ' + \
                             Program.OP_SYMBOLS[op_code] + \
                             " " + \
                             source + \
                             '[' + \
                             str(source_index) + \
                             ']'

        print(instruction_string)


    def execute(self, IP):
        '''Execute the program's instructions on some examplar,
        provided as input (IP) to this function.
        '''

        # Reset registers before performing computation
        self.registers[:] = 0

        for instruction in self.instructions:
            self.executeInstruction(instruction, IP)


    def executeInstruction(self, instruction, IP):

        # Extract instruction values
        mode, target_index, op_code, source_index = instruction

        # Perform modulo in case we are indexing into the smaller range
        source_index %= self._source_mod_value[mode]

        # Switch on mode (register mode vs input mode)
        source = None
        if mode == Program.REGISTER_MODE:
            source = self._registers
        else:
            source = IP

        # Switch on operation and execute
        if op_code == Program.ADDITION_OP:
            self._registers[target_index] = self._registers[target_index] + source[source_index]

        elif op_code == Program.SUBTRACTION_OP:
            self._registers[target_index] = self._registers[target_index] - source[source_index]

        elif op_code == Program.MULTIPLICATION_OP:
            self._registers[target_index] = self._registers[target_index] * source[source_index]

        elif op_code == Program.DIVISION_OP:
            if source[source_index] == 0:
                self._registers[target_index] = Program.SAFE_DIVISION_RESULT
            else:
                self._registers[target_index] = self._registers[target_index] / source[source_index]


    def createRandomInstruction(self):
        mode          = randint(0, Program.NUM_MODES)
        target_index  = randint(0, self._num_registers)
        op_code       = randint(0, Program.NUM_OP_CODES)
        source_index  = randint(0, self._max_source_range)

        return [mode, target_index, op_code, source_index]


    def copy(self, do_initialize_instructions=False, do_copy_instructions=False):
        '''Copy a program instance. Used in crossover and for creating populations.

        Params:
        do_initialize_instructions: Boolean determining whether the child initializes
        its own set of instructions. This is useful when a single instance of a Program
        is to be used as a template for many instances, ie. in population creation.

        do_copy_instructions: Boolean determining whether the child should receive
        a copy of its parent's instructions. Useful for crossover.
        '''
        p = Program(max_initial_instructions = self._max_initial_instructions,
                    num_registers            = self._num_registers,
                    num_inputs               = self._num_inputs,
                    mutation_rate            = self._mutation_rate,
                    max_num_instructions     = self._max_num_instructions,
                    num_classes              = self._num_classes,
                    initialize_instructions  = do_initialize_instructions)

        if do_copy_instructions:
            p.instructions = copy.deepcopy(self.instructions)

        return p


    def crossover(self, other):
        '''Canonical probabilistic crossover.'''

        self_num_instructions  = len(self.instructions)
        other_num_instructions = len(other.instructions)

        c1 = None
        c2 = None
        c1_instructions = None
        c2_instructions = None

        # With a probability of 10%, possibly avoid crossover and give
        # children exact copies of the parents' instructions.
        if uniform(0.0, 1.0) > 0.9:
            c1 = self.copy()
            c2 = other.copy()
            c1._instructions = copy.deepcopy(self._instructions)
            c2._instructions = copy.deepcopy(other._instructions)
            return c1, c2

        self_point1 = randint(0,               self_num_instructions)
        self_point2 = randint(self_point1 + 1, self_num_instructions + 1)

        other_point1 = randint(0,                other_num_instructions)
        other_point2 = randint(other_point1 + 1, other_num_instructions + 1)

        c1_instructions = self.instructions[0:self_point1] + \
                          other.instructions[other_point1:other_point2] + \
                          self.instructions[self_point2:]

        c2_instructions = other.instructions[0:other_point1] + \
                          self.instructions[self_point1:self_point2] + \
                          other.instructions[other_point2:]

        c1 = self.copy()
        c2 = other.copy()

        if len(c1_instructions) > self._max_num_instructions or \
           len(c2_instructions) > self._max_num_instructions:
            c1.instructions = copy.deepcopy(self.instructions)
            c2.instructions = copy.deepcopy(other.instructions)
        else:
            c1.instructions = copy.deepcopy(c1_instructions)
            c2.instructions = copy.deepcopy(c2_instructions)

        return c1, c2

    '''
    def mutate(self):
        if uniform(0.0, 1.0) > self._mutation_rate:
            return

        # Get random instruction
        instruction_index = randint(0, len(self.instructions))

        # Get random instruction component
        component_index = randint(0, Program.NUM_INSTRUCTION_COMPONENTS)

        # Figure out which component is being modified
        upper_bound = None
        if component_index == Program.MODE_INDEX:
            upper_bound = Program.NUM_MODES
        elif component_index == Program.TARGET_INDEX:
            upper_bound = self._num_registers
        elif component_index == Program.OP_CODE_INDEX:
            upper_bound = Program.NUM_OP_CODES
        else:
            upper_bound = self._max_source_range

        # Mutate component
        new_val = randint(0, upper_bound)
        while new_val == self.instructions[instruction_index][component_index]:
            new_val = randint(0, upper_bound)
        self.instructions[instruction_index][component_index] = new_val
    '''

    def mutate(self):
        #Canonical probabilistic mutation.

        num_mutated = 0

        for i in range(len(self.instructions)):

            # For each instruction, flip a coin weighted at the _mutation_rate.
            # If it comes up "unlikely heads", then mutate the instruction.
            if uniform(0.0, 1.0) > self._mutation_rate:
                continue

            num_mutated += 1

            # Get random instruction component
            component_index = randint(0, Program.NUM_INSTRUCTION_COMPONENTS)

            # Figure out which component is being modified and set the upper
            # bound on the RNG.
            upper_bound = None
            if component_index == Program.MODE_INDEX:
                upper_bound = Program.NUM_MODES
            elif component_index == Program.TARGET_INDEX:
                upper_bound = self._num_registers
            elif component_index == Program.OP_CODE_INDEX:
                upper_bound = Program.NUM_OP_CODES
            else:
                upper_bound = self._max_source_range

            # Mutate component
            new_val = randint(0, upper_bound)
            while new_val == self.instructions[i][component_index]:
                new_val = randint(0, upper_bound)
            self.instructions[i][component_index] = new_val

        return num_mutated

    def evaluate(self, X, y):
        '''Evaluate the current program on a labelled dataset. This function
        returns a *unitless* error value, ie. this value is NOT a percent or
        meaningful value. Rather: lower good, higher bad.

        Params:
        X: List of exemplars where each examplar is a list of floating point values.
        y: List of labels where each label is an integer that represents the class.
        '''
        error = 0

        for i, x in enumerate(X):
            self.execute(x)
            z = None
            if self._num_classes == 2:
                z = self.registers[0]
                error += logisticCrossEntropy(z, y[i])
            else:
                z = self.registers[0:self._num_classes]
                error += softmaxCrossEntropy(z, y[i])

        return error


    def accuracy(self, X, y):
        '''Evaluate the accuracy of the program on a labelled dataset.
        This function returns the fraction of examplars correctly labelled
        by the program.

        Params:
        X: List of exemplars where each examplar is a list of floating point values.
        y: List of labels where each label is an integer that represents the class.
        '''

        num_correct = 0
        num_samples = len(y)

        for i, x in enumerate(X):
            self.execute(x)
            z          = None
            prediction = None
            if self._num_classes == 2:
                z = self.registers[0]
                prediction = np.round(sigmoid(z))
            else:
                z = self.registers[0:self._num_classes]
                prediction = np.argmax(z)
            if prediction == y[i]:
                num_correct += 1

        return (num_correct/num_samples)


    def save(self, file_name):
        with open(file_name, 'wb') as fp:
            pickle.dump(p, fp)


    @staticmethod
    def load(file_name):
        p = None
        with open (file_name, 'rb') as fp:
            p = pickle.load(fp)
        return p


def tournamentSelection(population_size,
                        template_program,
                        halting_fitness,
                        max_num_generations,
                        X,
                        y,
                        display_fun=None):

    history = {
        'error': []
    }

    if display_fun == None:
        display_fun = print

    population = []
    for i in range(population_size):
        p = template_program.copy(do_initialize_instructions=True)
        population.append(p)

    best_fitness    = 1000000
    best_performer  = None

    for r in range(max_num_generations):

        # Get four random competitors
        competitor_indices = np.random.choice(range(len(population)), size=4, replace=False)
        competitors = [population[i] for i in competitor_indices]

        # Pit the competitors against each other
        # (evaluate their fitness)
        fitness = []
        for competitor in competitors:
            fitness.append(competitor.evaluate(X, y))

        # Rank the competitors
        fitness_indices = np.argsort(fitness)

        # Record best fitness and best performer
        if fitness[fitness_indices[0]] < best_fitness:
            best_fitness   = fitness[fitness_indices[0]]
            best_performer = population[fitness_indices[0]]

            if best_fitness < halting_fitness:
                display_fun("Round " + str(r) + " - Achieved fitness of " + str(best_fitness) + " < " + str(halting_fitness))
                break

        history['error'].append(best_fitness)

        # Grab out the two top players as parents
        p1, p2 = competitors[fitness_indices[0]], competitors[fitness_indices[1]]

        c1, c2 = p1.crossover(p2)
        c1.mutate()
        c2.mutate()

        population[competitor_indices[fitness_indices[2]]] = c1
        population[competitor_indices[fitness_indices[3]]] = c2

        if display_fun != None:
            display_fun("Round " + str(r) + " - Error " + str(best_fitness))

    return best_performer, history


def breederSelection(population_size,
                     template_program,
                     halting_fitness,
                     max_num_generations,
                     gap_percent,
                     X,
                     y,
                     display_fun=None):

    history = {
        'error': []
    }

    if display_fun == None:
        display_fun = print

    population = []
    for i in range(population_size):
        p = template_program.copy(do_initialize_instructions=True)
        population.append(p)
    population = np.array(population)

    best_fitness        = 1000000
    best_performer      = None
    num_gap_individuals = round(gap_percent * len(population))

    for r in range(max_num_generations):

        # Pit the competitors against each other (evaluate their fitness)
        fitness = []
        for p in population:
            fitness.append(p.evaluate(X, y))
        fitness = np.array(fitness)

        # Rank the competitors
        fitness_indices = np.argsort(fitness)

        # Record best fitness and best performer
        history['error'].append(fitness[fitness_indices[0]])
        if fitness[fitness_indices[0]] < best_fitness:
            best_fitness   = fitness[fitness_indices[0]]
            best_performer = population[fitness_indices[0]]

            if best_fitness < halting_fitness:
                display_fun("Round " + str(r) + " - Achieved fitness of " + str(best_fitness) + " < " + str(halting_fitness))
                break

        # Remove num_gap_individuals
        population = population[fitness_indices[0:-num_gap_individuals]]

        children = []
        while len(children) < num_gap_individuals:

            # Get two random individual
            parent_indices = np.random.choice(range(len(population)), size=2, replace=False)
            p1, p2 = population[parent_indices]

            c1, c2 = p1.crossover(p2)
            c1.mutate()
            c2.mutate()

            children.append(c1)
            children.append(c2)

        children = np.array(children)
        population = np.concatenate((population, children))

        if display_fun != None:
            display_fun("Round " + str(r) + " - Error " + str(best_fitness))

    return best_performer, history
