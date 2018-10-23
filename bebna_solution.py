
import numpy as np
import tensorflow as tf

##from tensorflow.keras.engine.topology import Layer
##from tensorflow.keras import backend as K

from retro_contest.local import make


import gym_remote.exceptions as gre
import gym_remote.client as grc

#from hashlib import sha1
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import imagehash
#import json


import NEATish
import copy

import asyncio



class bebna:
    def __init__(self, initial_random_actions=256, num_actions=12,
            num_frames_before_show=100, encoding_layer_depth=560):
        self.initial_random_actions = initial_random_actions
        self.num_frames_before_show = num_frames_before_show
        self.current_frame = 0
        self.num_actions = num_actions
        self.num_learning_actions = 9
        self.encoding_layer_depth = encoding_layer_depth

        self.current_fitness = 0.0
        self.active_reward_fitness = 0.0


        self.b_action = 0
        self.a_action = 1
        self.mode_action = 2
        self.start_action = 3
        self.up_action = 4
        self.down_action = 5
        self.left_action = 6
        self.right_action = 7
        self.c_action = 8
        self.y_action = 9
        self.x_action = 10
        self.z_action = 11

        self.empty_train = 0
        self.left_train = 1
        self.right_train = 2
        self.left_down_train = 3
        self.right_down_train = 4
        self.down_b_train = 5
        self.left_b_train = 6
        self.right_b_train = 7
        self.b_train = 8
        self.down_train = 9
        '''
        self.opposites = {
                self.left_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_train:[self.left_train,
                    self.left_down_train, self.right_b_train],
                self.left_down_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_down_train:[self.left_train,
                    self.left_down_train, self.left_b_train],
                self.down_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.down_b_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.left_b_train:[self.right_train, self.right_b_train,
                    self.down_b_train, self.right_down_train],
                self.right_b_train:[self.left_train, self.left_b_train,
                    self.down_b_train, self.left_down_train],
                self.b_train:[self.left_train, self.right_train,
                    self.left_down_train, self.right_down_train,
                    self.down_train, self.down_b_train, self.left_b_train,
                    self.right_b_train],
                self.empty_train:[self.right_train,
                    self.right_down_train,
                    self.down_train, self.down_b_train,
                    self.right_b_train,
                    self.b_train]
        }
        '''
        self.opposites = {
                self.left_train:[self.right_train,
                    self.right_down_train, self.right_b_train],
                self.right_train:[self.left_train,
                    self.left_down_train, self.right_b_train],
                self.left_down_train:[self.right_train,self.right_b_train],
                self.right_down_train:[self.left_train,self.left_b_train],
                self.down_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.down_b_train:[self.left_train, self.right_train,
                    self.left_b_train, self.right_b_train],
                self.left_b_train:[self.right_train, self.right_b_train, self.right_down_train],
                self.right_b_train:[self.left_train, self.left_b_train, self.left_down_train],
                self.b_train:[self.left_train, self.right_train,
                    self.left_down_train, self.right_down_train,
                    self.down_b_train, self.left_b_train,
                    self.right_b_train],
                self.empty_train:[self.right_train,
                    self.right_down_train,
                    self.down_train, self.down_b_train,
                    self.right_b_train,
                    self.b_train]
        }



    def setup(self, env, observation_hashes=None):
        self.env = env

        self.NEAT = NEATish.Population(self.encoding_layer_depth, self.num_learning_actions, 'fortytwo', 60, 250, 5, 5)

        self.last_action = self.env.action_space.sample()
        self.last_hash = None
        self.current_random_actions = 0

        self.observations = np.zeros((32, 224, 320, 3))
        self.num_observations = 0
        self.unique_observations = np.zeros((16,224,320, 3))
        ## How different the unique observations are to each other.
        ## We want high numbers here, rather than low ones.
        self.similarity = np.zeros((16, 16))
        self.uniqueness = np.zeros(16)
        self.uniqueness_check = np.zeros(16)
        if observation_hashes:
            self.observation_hashes = observation_hashes
        else:
            self.observation_hashes = {}
        self.config = tf.ConfigProto()
        self.session = tf.Session(config=self.config)
        self.encoder_model = None
        self.decoder_model = None
        self.actuator_model = None
        self.make_encoder_decoder()
        ## Stuff for showing how the encoder is doing.
        self.img_height = 224
        self.img_width = 320
        self.double_height = 448
        self.encoder_images = np.zeros((self.double_height, self.img_width, 3))
        plt.ion()
        self.plot_title = plt.title("Encoding_Figure")
        self.fig, self.ax = plt.subplots(1,1)
        self.image = self.ax.imshow(self.encoder_images[:, :, :], animated=True)
        self.fig.canvas.draw()

        self.initial_exploration_index = 0
        self.initial_exploration_counter = 0
        self.initial_exploration_list = np.arange(10)
        np.random.shuffle(self.initial_exploration_list)

        self.num_unique_frames = 0

        self.map_tracker = MapTracker()
        self.keyframe_counter = 0

        self.last_reward = 0
        self.current_path = []

        self.new_problem_completed = False
        self.processing_old_paths = False

        self.population_done = False

        self.observations_added = False
        self.current_observation = 0

        self.encoded_frames = []




    def convert_to_train_space(self, data):
        ## Data is 12 units long.
        ## Map from base action space to reasonable action space.
        new_data = [0]*10
        if data[self.down_action] == 1:
            if data[self.b_action] == 1:
                new_data[self.down_b_train] = 1
            elif data[self.left_action] == 1:
                new_data[self.left_down_train] = 1
            elif data[self.right_action] == 1:
                new_data[self.right_down_train] = 1
            else:
                new_data[self.down_train] = 1
        elif data[self.b_action] == 1:
            if data[self.right_action] == 1:
                new_data[self.right_b_train] = 1
            elif data[self.left_action] == 1:
                new_data[self.left_b_train] = 1
            else:
                new_data[self.b_train] = 1
        elif data[self.left_action] == 1:
            new_data[self.left_train] = 1
        elif data[self.right_action] == 1:
            new_data[self.right_train] = 1
        else:
            new_data[self.empty_train] = 1
        return new_data

    def convert_to_action_space(self, data):
        ## Data is one integer.
        new_data = [0]*12
        #print(data)
        if data == self.empty_train:
            pass
        if data == self.left_train:
            new_data[self.left_action] = True
        if data == self.right_train:
            new_data[self.right_action] = True
        if data == self.left_down_train:
            new_data[self.left_action] = True
            new_data[self.down_action] = True
        if data == self.right_down_train:
            new_data[self.right_action] = True
            new_data[self.down_action] = True
        if data == self.down_train:
            #print('Oops, down...')
            new_data[self.down_action] = True
        if data == self.down_b_train:
            new_data[self.down_action] = True
            new_data[self.b_action] = True
        if data == self.b_train:
            new_data[self.b_action] = True
        return new_data

    def process_data(self, obs, info=None, f=None, reward=None):
        genome_processed = False
        generation_processed = False
        restart_population = False
        if self.map_tracker.keystone_image is None:
            if self.keyframe_counter > 3:
                self.map_tracker.keystone_image = obs
                self.last_obs = obs
            else:
                self.keyframe_counter += 1
            temp_reward = 0
        else:
            if len(self.NEAT.previous_path) > 0:
                self.processing_old_paths = True
            else:
                self.processing_old_paths = False
            temp_reward, new_problem_completed, restart_population = self.map_tracker.calculate_reward(self.last_obs, obs, reward, self.current_path, self.NEAT, self.processing_old_paths, info)
            
            if new_problem_completed:
                #genome_processed = True
                self.new_problem_completed = True
        ## This was the old hash, we are now using distance hashes.
        #temp_hash = sha1(obs).hexdigest()
        ## New hashing method.
        temp_hash = imagehash.whash(Image.fromarray(obs, mode='RGB'))
        if self.last_hash:
            if temp_hash - self.last_hash < 10:
                temp_hash = self.last_hash
                #self.NEAT.extra_frames_earned -= 1
        ## We want to do random things for a bit, but not too long. Then move
        ## on to actually predicting data.

        ## First, encode the frame.
        encoded_frame = self.encoder_model.predict(np.expand_dims(obs, axis=0))
        encoded_frame *= (1.0/1000.0)
        if self.map_tracker.last_encoded_frame is None:
            self.map_tracker.last_encoded_frame = encoded_frame
        self.map_tracker.encoded_frame = encoded_frame
        ##print(self.map_tracker.last_encoded_frame - self.map_tracker.encoded_frame)
        #print("np max",np.amax(encoded_frame))
        #print("np min",np.amin(encoded_frame))




        #last_img = self.observations[1]
        #current_image = self.observations[0]
        #print("Frame reward: {}".format(temp_reward))
        self.current_fitness += temp_reward
        self.active_reward_fitness += temp_reward

        #print("Total reward: {}".format(self.current_fitness))
        current_drop = math.log(self.NEAT.current_generation + 1) / 100.0
        temp_fitness_check = 100000.0 / (current_drop + 1.0) + 1
        if self.active_reward_fitness > temp_fitness_check:
            self.NEAT.extra_frames_earned += (self.active_reward_fitness // temp_fitness_check) * 8
            self.active_reward_fitness %= temp_fitness_check
            #print("Nuber frames added: {}".format(self.NEAT.extra_frames_earned))


        '''
        if temp_hash in self.observation_hashes:
            self.current_fitness += 1000 / self.observation_hashes[temp_hash]
            self.active_reward_fitness += 1000 / self.observation_hashes[temp_hash]
            self.observation_hashes[temp_hash] += 1
        else:
            self.current_fitness += 1000.0
            self.active_reward_fitness += 1000.0
            self.num_unique_frames += 1
            print("Number of unique frames: {}".format(self.num_unique_frames))
            self.observation_hashes[temp_hash] = 1

        if self.active_reward_fitness > 1000.0:
            self.NEAT.extra_frames_earned += (self.active_reward_fitness // 1000) * 10
            self.active_reward_fitness %= 1000.0
        '''




        if self.current_random_actions < 0: #self.initial_random_actions:
            current_action = self.initial_exploration_index
            output_data = self.convert_to_action_space(current_action)
            #output_data = self.env.action_space.sample()
            self.current_random_actions += 1
            self.initial_exploration_counter += 1
            if self.initial_exploration_counter > 14:
                self.initial_exploration_counter = 0
                self.initial_exploration_index += 1
                if self.initial_exploration_index > self.initial_exploration_list.size:
                    self.initial_exploration_index = 0
            self.current_fitness = 0.0
            self.active_reward_fitness = 0.0
        else:
            ## TODO: Actually do something :P

            output = self.NEAT.process_data(encoded_frame[0])
            ## Check if we need to start a new generation.
            self.NEAT.check_location(self.map_tracker.current_x_y_coords)
            if self.NEAT.genome_processed:
                genome_processed = True
                self.reset()
            if self.NEAT.generation_processed:
                generation_processed = True
                self.NEAT.process_new_generation()
                #print('Generation processed.')

                #input('Generation processed...')


            ## Start passing the encoded_frame into the NEAT implementation.
            self.NEAT.update_position()

            output_data = np.argmax(output)
            output_data = self.convert_to_action_space(output_data)
            #print(encoded_frame)
            #print(output_data)
            #input('...')

            #if np.array_equal(self.last_action, np.array([1, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0])):
            #    output_data = np.array([0, 0, 0, 0, 0, 0, 0,
            #                   1, 0, 0, 0, 0])
            #else:
            #    output_data = np.array([1, 0, 0, 0, 0, 0, 0,
            #                   1, 0, 0, 0, 0])


        ## Update the network, save the output from this, and return the new set
        ## of instructions.
        ##print("output_data: {}".format(output_data))
        self.last_action = output_data
        self.last_hash = temp_hash
        self.last_obs = obs
        self.add_observation()
        ##self.train_auto_encoder()
        self.map_tracker.last_encoded_frame = encoded_frame
        if self.new_problem_completed and generation_processed:
            self.new_problem_completed = False
            genome_processed = True
            #if self.NEAT.best_genome is None:
            #    print(self.NEAT.species_counter)
            #    self.NEAT.best_genome = self.NEAT.species_list[self.NEAT.species_counter].genomes[self.NEAT.genome_counter]
            temp_genome = copy.deepcopy(self.NEAT.best_genome)
            old_path_info = (temp_genome, (self.NEAT.best_genome_end[0], self.NEAT.best_genome_end[1]))
            self.current_path.append(old_path_info)
            ## Need to reset the NEAT population and add on the current path data.
            current_path_copy = copy.deepcopy(self.current_path)
            self.NEAT.setup(current_path_copy)
            self.population_done = True
            #input("Adding new genome to path...")

        if restart_population:
            self.reset_population()
            genome_processed = True
            #generation_processed = True
        return output_data, genome_processed

    def train_auto_encoder(self):
        ## Only using each pair every other set of calculations should be fine.
        ## Each should get eight runs still and this way you don't have the same
        ## pairs every time, just every other.
        #self.add_observation()
        inputs = self.observations[0::2]## the last 16 frames, paired off temporally
        outputs = self.observations[1::2]## the last 16 frames, paired off temporally
        zero_input = np.zeros((16,560))
        self.training_model.train_on_batch([inputs, inputs, zero_input], [inputs, inputs, zero_input])
        ##self.add_observation()
        #input()

    def add_observation(self):
        
        if self.current_observation < 32:
            np.roll(self.observations, 1, axis=0)
            self.observations[0] = self.last_obs
            self.current_observation += 1
        else:
            self.current_observation = 0
            self.train_auto_encoder()

        

        
        predicted_frames = self.visualization_model.predict(np.expand_dims(self.observations[0], axis=0))
       
        self.animate_data(self.observations[0], predicted_frames[0])
        '''
        if not self.observations_added:
            for i in range(32):
                self.observations[i] = self.last_obs
            self.observations_added = True
        else:
            np.roll(self.observations, 1, axis=0)
            self.observations[0] = self.last_obs
        '''
        

    def animate_data(self, screen_frame, predicted_frame):
        #print(screen_frame[0][0][0], screen_frame[0][0][1], screen_frame[0][0][2])
        #print(np.amax(screen_frame))
        #screen_frame = screen_frame[:,:,::-1]
        #predicted_frame = predicted_frame[:,:,::-1]
        ## Set top half of the encoder images to be the input frame.
        self.encoder_images[:self.img_height,:,:] = screen_frame
        self.encoder_images[self.img_height:, :,:] = predicted_frame
        self.image.set_data(self.encoder_images[:,:,:])
        self.fig.canvas.draw()

    def reset(self):
        #print("reset!")
        #print("Sonic's reward: {}".format(self.current_fitness))
        if self.current_fitness < 0.0:
            self.current_fitness = 0.0
        self.NEAT.apply_reward(self.current_fitness)
        self.current_fitness = 0.0
        self.active_reward_fitness = 0.0
        self.keyframe_counter = 0
        self.map_tracker.reset()
        self.observation_hashes = {}
        self.num_unique_frames = 0
        self.NEAT.extra_frames_earned = 0
        self.NEAT.previous_path = copy.deepcopy(self.NEAT.previous_path_master)

    def reset_population(self):
        self.reset()
        self.NEAT.extra_frames_earned = 0
        current_path_copy = copy.deepcopy(self.current_path)
        self.NEAT.setup(current_path_copy)



    def make_encoder_decoder(self):
        try:
            #self.encoder_model = tf.keras.models.load_model('models/encoder_model')
            self.encoder_frozen_model = tf.keras.models.load_model('models/encoder_model.h5')
            #for layer in self.encoder_frozen_model.layers:
            #    layer.trainable=False

            #self.decoder_model = tf.keras.models.load_model('models/decoder_model')
            self.decoder_frozen_model = tf.keras.models.load_model('models/decoder_model.h5')
            #for layer in self.decoder_frozen_model.layers:
            #    layer.trainable=False

            #self.puzzle_solver_model = tf.keras.models.load_model('models/puzzle_model.h5')
            #self.decoder_eval_model = tf.keras.models.load_model('models/decoder_eval_model.h5')

            #print(self.encoder_frozen_model.summary())

        except:
            #input('...')
            print("Error loading models.")
            self.encoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                    )
            self.encoder_convolution_0 = tf.keras.layers.Conv2D(64,
                    kernel_size=(4, 4),strides=(4,4), padding='valid')(self.encoder_input)
            self.encoder_convolution_1 = tf.keras.layers.Conv2D(128,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_0)
            self.encoder_convolution_2 = tf.keras.layers.Conv2D(256,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_1)
            self.encoder_convolution_3 = tf.keras.layers.Conv2D(8,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.encoder_convolution_2)
            self.encoder_encoding_layer = tf.keras.layers.Flatten()(
                    self.encoder_convolution_3)
            #self.encoder_encoding_layer = tf.keras.layers.Dense(256,
            #        )(self.encoder_flat)
            self.encoder_model = tf.keras.models.Model(
                    self.encoder_input, self.encoder_encoding_layer)

            self.image_1_input = tf.keras.layers.Input(shape=(224, 320, 3,))
            self.encoded_image_1 = self.encoder_model(self.image_1_input)
            self.image_2_input = tf.keras.layers.Input(shape=(224, 320, 3,))
            self.encoded_image_2 = self.encoder_model(self.image_2_input)

            self.zero_layer = tf.keras.layers.Input(shape=(560,))
            self.encoded_image_2_negated = tf.keras.layers.Lambda(lambda x: -x)(self.encoded_image_2)
            self.encoded_difference = tf.keras.layers.Add()([self.encoded_image_1, self.encoded_image_2_negated])
            ##self.encoded_absolute_difference = tf.keras.backend.abs(self.encoded_difference)

            self.decoder_input = tf.keras.layers.Input(shape=(560,),
                    name='decoder_encoder_input')
            self.decoder_reshaped = tf.keras.layers.Reshape(
                    target_shape=(7, 10, 8))(self.decoder_input)
            self.decoder_deconvolution_1 = tf.keras.layers.Conv2DTranspose(2048,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_1_2 = tf.keras.layers.Conv2DTranspose(1024,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_2 = tf.keras.layers.Conv2DTranspose(512,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1)
            self.decoder_deconvolution_2_2 = tf.keras.layers.Conv2DTranspose(256,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1_2)
            self.decoder_tower_output = tf.keras.layers.concatenate([self.decoder_deconvolution_2, self.decoder_deconvolution_2_2])
            self.decoder_deconvolution_3 = tf.keras.layers.Conv2DTranspose(128,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_tower_output)
            self.decoder_output = tf.keras.layers.Conv2DTranspose(3,
                    kernel_size=(4,4), strides=(4,4), padding='valid',
                    activation='sigmoid',
                    name="decoder_output")(self.decoder_deconvolution_3)

            self.decoder_model = tf.keras.models.Model([self.decoder_input],
                    [self.decoder_output])


            self.visualization_input = tf.keras.layers.Input(shape=(224, 320, 3,))
            self.visualization_encoded = self.encoder_model(self.visualization_input)
            self.visualization_output = self.decoder_model(self.visualization_encoded)
            self.visualization_model = tf.keras.models.Model(self.visualization_input, self.visualization_output)
            self.visualization_model.compile(optimizer="adam", loss="mse")

            
            self.image_1_decoded = self.decoder_model(self.encoded_image_1)
            self.image_2_decoded = self.decoder_model(self.encoded_image_2)
            self.training_model = tf.keras.models.Model([self.image_1_input, self.image_2_input, self.zero_layer],
                    [self.image_1_decoded, self.image_2_decoded, self.encoded_difference])
            self.training_model.compile(optimizer="adam", loss="mse", loss_weights=[0.3, 0.3, 0.3])
            ## Decoder
            ## The encoder segment should be trainable while we are training the
            ## autoencoder so we do not deactivate them here.
            ## Secondary path is based on: https://arxiv.org/pdf/1603.09246.pdf
            ## Secondary path for the decoder. Plan of action:
            ## We are going to slice the image up into units of h:112 w:160
            ## This makes a 4 by 4 grid of the image.
            ## Make a list from zero to 3.
            ## Randomly shuffle that list.

            '''
            self.decoder_sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.1,
                    decay=0.1)

            #self.decoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
            #        name='decoder_encoder_input')
            self.decoder_encoder_layer = self.encoder_model(self.decoder_input)

            ## This next line is used if we ever want to be able to run the
            ## decoder by itself. Probably not something we need but it is
            ## there anyways.
            ##self.decoder_input = tf.layers.Input(128, name="decoder_input")
            self.decoder_reshaped = tf.keras.layers.Reshape(
                    target_shape=(7, 10, 8))(self.decoder_encoder_layer)
            self.decoder_deconvolution_1 = tf.keras.layers.Conv2DTranspose(2048,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_1_2 = tf.keras.layers.Conv2DTranspose(1024,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_reshaped)
            self.decoder_deconvolution_2 = tf.keras.layers.Conv2DTranspose(512,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1)
            self.decoder_deconvolution_2_2 = tf.keras.layers.Conv2DTranspose(256,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_deconvolution_1_2)
            self.decoder_tower_output = tf.keras.layers.concatenate([self.decoder_deconvolution_2, self.decoder_deconvolution_2_2])
            self.decoder_deconvolution_3 = tf.keras.layers.Conv2DTranspose(128,
                    kernel_size=(2,2), strides=(2,2), padding='valid',
                    activation='relu')(self.decoder_tower_output)
            self.decoder_output = tf.keras.layers.Conv2DTranspose(3,
                    kernel_size=(4,4), strides=(4,4), padding='valid',
                    activation='sigmoid',
                    name="decoder_output")(self.decoder_deconvolution_3)
            #self.decoder_dense_output = tf.keras.layers.Dense(215040, activation='sigmoid')(self.decoder_deconvolution_2)
            #self.decoder_output = tf.keras.layers.Reshape(target_shape=(224, 320, 3),name="decoder_output")(self.decoder_dense_output)

            self.decoder_model = tf.keras.models.Model([self.decoder_input, self.zero_layer],
                    [self.decoder_output, self.encoded_absolute_difference])

            self.two_tower_decoder_model = tf.keras.models.Model
            self.decoder_model.compile(optimizer='adam', loss='mse', loss_weights=[0.5, 0.5])
            #print(self.decoder_model.summary())

            #self.puzzle_plus_decoder_model = tf.keras.models.Model(
            #        inputs=[self.decoder_input, self.puzzle_solver_input],
            #        outputs=[self.decoder_output, self.puzzle_solver_output])
            #self.puzzle_plus_decoder_model.compile(optimizer='sgd',
            #        loss={'decoder_output':'mse', 'puzzle_output':'categorical_crossentropy'},
            #        loss_weights={'decoder_output':0.45, 'puzzle_output':3.0})

            ## Actuator
            ## We make sure here that the encoder is not trainable when we call
            ## the actuator.
            self.encoder_convolution_0.trainable = False
            self.encoder_convolution_1.trainable = False
            self.encoder_convolution_2.trainable = False
            self.encoder_convolution_3.trainable = False
            self.encoder_encoding_layer.trainable = False


            ## For looking at decoder output.
            #self.decoder_decoding_layer.trainable = False
            self.decoder_reshaped.trainable = False
            self.decoder_deconvolution_1.trainable = False
            self.decoder_deconvolution_2.trainable = False
            self.decoder_deconvolution_3.trainable = False

            self.decoder_output.trainable = False
            self.decoder_eval_model = tf.keras.models.Model([self.decoder_input],
                    [self.decoder_output])
            self.decoder_eval_model.compile(optimizer=self.decoder_sgd, loss='mse')
            ##print(self.decoder_model.summary())
            ## Also make and compile the encoder model here so it can be used
            ## in a frozen manner.
            #self.encoder_model = tf.keras.models.Model(
            #        inputs=[self.encoder_input],
            #        outputs=[self.encoder_encoding_layer])

            #print(self.encoder_model.summary())

            #self.encoder_sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0,
            #        decay=0.0, nesterov=True)
            '''
            self.encoder_model.compile(optimizer='adam', loss='mse')



        ## Predictor.
        ## LSTM input is number in encoded layer(256) + number of actions(8)
        self.lstm_input_size = self.encoding_layer_depth + 8
        self.predictor_input = tf.keras.layers.Input(batch_shape=(self.lstm_input_size, 1, 1))
        self.predictor_LSTM = tf.keras.layers.LSTM(128, bias_initializer='glorot_uniform', stateful=True, return_state=True)
        self.predictor_LSTM_output, self.predictor_LSTM_state_h, self.predictor_LSTM_state_c = self.predictor_LSTM(self.predictor_input)
        self.predictor_fc_1 = tf.keras.layers.Dense(64)(self.predictor_LSTM_output)
        self.predictor_fc_2 = tf.keras.layers.Dense(self.encoding_layer_depth)(self.predictor_LSTM_output)
        #self.predictor_MDN = MDN(self.lstm_input_size, 8)
        self.predictor_model = tf.keras.models.Model([self.predictor_input], [self.predictor_fc_2, self.predictor_LSTM_state_h])
        #self.predictor_optimizer = Adam(lr=0.001)
        #self.predictor_model = tf.keras.models.Model([self.predictor_input], [self.predictor_MDN])
        #self.predictor_model.compile(loss=mdn_loss(), optimizer=self.predictor_optimizer)
        #self.predictor_model.compile(loss=mdn_loss(), optimizer='adam')
        self.predictor_model.compile(loss="logcosh", optimizer='adam')


        ## Actuator code.
        ## Use NEAT rather than tensorflow for this.
        '''
        self.actuator_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                name='actuator_input')
        self.actuator_encoded = self.encoder_frozen_model(self.actuator_input)
        self.actuator_dense_1 = tf.keras.layers.Dense(64)(
                self.actuator_encoded)
        self.actuator_dense_2 = tf.keras.layers.Dense(32)(
                self.actuator_dense_1)
        self.actuator_dense_3 = tf.keras.layers.Dense(16)(
                self.actuator_dense_2)
        self.actuator_output = tf.keras.layers.Dense(10,
                name="actuator_output",
                activation="softmax")(self.actuator_dense_3)
        self.actuator_model = tf.keras.models.Model([self.actuator_input],
                [self.actuator_output])

        self.actuator_model.compile(optimizer='adam',
                loss='categorical_crossentropy')
        '''

        self.NEAT.setup()


        with self.session as sess:
            #tf.initialize_all_variables().run()
            tf.global_variables_initializer().run()

'''
## MDN Stuff - https://github.com/yanji84/keras-mdn

def get_mixture_coef(output, numComonents=24, outputDim=1):
    out_pi = output[:,:numComonents]
    out_sigma = output[:,numComonents:2*numComonents]
    out_mu = output[:,2*numComonents:]
    out_mu = tf.keras.backend.reshape(out_mu, [-1, numComonents, outputDim])
    out_mu = tf.keras.backend.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = tf.keras.backend.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = tf.keras.backend.exp(out_pi)
    normalize_pi = 1 / tf.keras.backend.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = tf.keras.backend.exp(out_sigma)
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = tf.keras.backend.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -tf.keras.backend.square(result)/2
    result = tf.keras.backend.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = tf.keras.backend.prod(result, axis=[0])
    return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = tf.keras.backend.sum(result, axis=1, keepdims=True)
    result = -tf.keras.backend.log(result + 1e-8)
    return tf.keras.backend.mean(result)

def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss

class MDN(tf.keras.layers.Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MDN, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = tf.keras.backend.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = tf.keras.backend.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = tf.keras.backend.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = tf.keras.backend.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.Wh) + self.bh)
        output = tf.keras.backend.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, inputShape):
        return (inputShape[0], self.outputDim)

'''

## New MDN:

class MapTracker:
    def __init__(self, keystone_image=None, image_height=56, image_width=80):
        self.last_encoded_frame = None
        self.keystone_image = keystone_image
        self.current_x_y_coords = (0.0, 0.0)
        self.reward_for_pixel = 10.0
        ## First channel is for checking for new pixels. Second channel is for dealing with checking
        ## to see if it is time to start a new population.
        self.pixels_seen = np.zeros(shape=(10001,10001,2), dtype=int)
        self.last_reward = 0
        self.world_reward = 0.0
        self.frames_in_old_area = 0

        self.image_height = image_height
        self.image_width = image_width
        ## Put all the initial pixels in the dictionary.
        #print(image_height+10000 // 2)
        for i in range(image_height // 2):
            for j in range(image_width // 2):
                self.pixels_seen[j + 5000][i + 5000][0] = 10
                self.pixels_seen[-j + 5000][i + 5000][0] = 10
                self.pixels_seen[j + 5000][-i + 5000][0] = 10
                self.pixels_seen[j + 5000][i + 5000][0] = 10


    def reset(self):
        #print('Last reward: {}.'.format(self.world_reward))
        #if self.world_reward > 8000:
        #    input('WE DID IT!...')
        self.last_reward = 0
        self.keystone_image = None
        self.current_x_y_coords = (0.0, 0.0)
        self.pixels_seen.fill(0)
        self.world_reward = 0.0
        self.frames_in_old_area = 0
        for i in range(self.image_height// 2):
            for j in range(self.image_width // 2):
                self.pixels_seen[j + 5000][i + 5000][0] = 10
                self.pixels_seen[-j + 5000][i + 5000][0] = 10
                self.pixels_seen[j + 5000][-i + 5000][0] = 10
                self.pixels_seen[j + 5000][i + 5000][0] = 10

    def calculate_reward(self, last_img, new_img, world_reward, current_path, NEAT, processing_old_paths=False, info=None):
        reward_changed = False
        restart_population = False
        self.world_reward += world_reward
        #print(self.world_reward)
        if self.world_reward is not None:
            if abs(self.world_reward - self.last_reward) > 500:
                ## This means that we might have progressed an interesting amount.
                #input("Last reward: {}, Reward: {}...".format(self.last_reward, self.world_reward))
                self.last_reward = self.world_reward
                reward_changed = True
        if self.keystone_image is None:
            self.keystone_image = new_img

        ## Scale the images down.
        last_img_resized = cv2.resize(last_img, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        new_img_resized = cv2.resize(new_img, dsize=(self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        #self.save_image(last_img, "last_img")
        #self.save_image(new_img, "new_img")
        #input("...")
        if imagehash.whash(Image.fromarray(new_img_resized, mode='RGB')) == imagehash.whash(Image.fromarray(self.keystone_image, mode='RGB')):
            self.current_x_y_coords = (0.0, 0.0)
            new_x = 0
            new_y = 0
        else:
            if info is not None:
                new_x = info['x'] / 4
                new_y = info['y'] / 4
            else:
                #print(last_img_resized.dtype, new_img_resized.dtype)
                try:
                    transform_matrix = cv2.estimateRigidTransform(new_img.astype('uint8'), last_img.astype('uint8'), False)
                    #print(last_img_resized)
                    #print(transform_matrix)
                    #x_shift = transform_matrix[0][2]
                    x_shift = world_reward / 4.0
                    y_shift = transform_matrix[1][2] / 4.0
                except:
                    #input("... transform is none")
                    return 0.0, False, restart_population
                #if x_shift < 1.0 and x_shift > -1.0:
                #    x_shift = 0
                if y_shift < 1.0 and y_shift > -1.0:
                    y_shift = 0
                    
                #if abs(y_shift) > 30:
                #    print(y_shift)
                #    return -300000.0, False
                #    #input('y_shift...')
                new_x = self.current_x_y_coords[0] + x_shift
                new_y = self.current_x_y_coords[1] + y_shift
            self.current_x_y_coords = (new_x, new_y)
        x_shift_int = int(math.floor(new_x) - self.image_width / 2)
        y_shift_int = int(math.floor(new_y) - self.image_height / 2)
        if x_shift_int > 4000:
            x_shift_int = 4000
        if x_shift_int < -4000:
            x_shift_int = -4000
        if y_shift_int > 4000:
            y_shift_int = 4000
        if y_shift_int < -4000:
            y_shift_int = -4000


        reward = 0.0
        in_old_area = False
        in_new_area = False
        stuck = False
        for i in range(self.image_height):
            for j in range(self.image_width):
                if processing_old_paths:
                    self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][0] += 1
                    self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][1] = 1
                    if self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][0] > 200:
                        stuck = True
                    self.last_reward = self.world_reward
                    #input('...')
                else:
                    self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][0] += 1
                    if self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][1] == 1:
                        #input('...')
                        self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][0] += 15000
                        in_old_area = True
                        self.frames_in_old_area += 1
                    else:
                        in_new_area = True
                        reward += self.reward_for_pixel / self.pixels_seen[j + 5000 + x_shift_int][i + 5000 + y_shift_int][0]
        if stuck:
            self.unstick(current_path, NEAT)
            restart_population = True
        if self.frames_in_old_area > 15:
            reward -= 500
        #print("Reward: {}.".format(reward))
        #print(x_shift_int, y_shift_int)

        new_problem_completed = False
        if not processing_old_paths and not in_old_area and reward_changed:
            #input('{}, {}...'.format(self.last_reward, world_reward))
            ## We are learning a new network(not replaying an old one), we don't see any of the old parts of the map,
            ## and we found a reward change area. Therefore we have probably solved part of the problem and should
            ## move on to try and solve a new part.

            ## This means that we need to stop learning on the current population and start a new one.
            ## Add the current genome to the list of path genomes.
            #input("Last reward: {}. Current reward: {}.".format(self.last_reward, self.world_reward))
            self.last_reward = self.world_reward
            #new_problem_completed = True

        #print("X shift: {}, Y shift: {}".format(self.current_x_y_coords[0], self.current_x_y_coords[1]))
        return reward, new_problem_completed, restart_population

    def unstick(self, current_path, NEAT):
        #print("Path old length: {}".format(len(current_path)))
        #print("Are we stuck?")

        ## Set the new boundary here.
        current_path[:] = current_path[:NEAT.path_position + 1]
        old_path_info = (current_path[-1][0], (self.current_x_y_coords[0], self.current_x_y_coords[1]))
        current_path[-1] = copy.deepcopy(old_path_info)
        #NEAT.previous_path = NEAT.previous_path[:1]
        #NEAT.previous_path[0] = copy.deepcopy(old_path_info)
        #NEAT.previous_path_master = NEAT.previous_path_master[:NEAT.path_position]
        #NEAT.previous_path_master[-1] = copy.deepcopy(old_path_info)
        #restart_population = True
        #input("Path length new: {}".format(len(current_path)))


def main():
    '''
    envs = [('SonicTheHedgehog-Genesis','GreenHillZone.Act',('1','3')),
            ('SonicTheHedgehog-Genesis','LabyrinthZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','MarbleZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','ScrapBrainZone.Act',('2')),
            ('SonicTheHedgehog-Genesis','SpringYardZone.Act',('2','3')),
            ('SonicTheHedgehog-Genesis','StarLightZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','AquaticRuinZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','CasinoNightZone.Act',('1')),
            ('SonicTheHedgehog2-Genesis','ChemicalPlantZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','EmeraldHillZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','HillTopZone.Act',('1')),
            ('SonicTheHedgehog2-Genesis','MetropolisZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','MysticCaveZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','OilOceanZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','AngelIslandZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','CarnivalNightZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','DeathEggZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','FlyingBatteryZone.Act',('1')),
            ('SonicAndKnuckles3-Genesis','HydrocityZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','IcecapZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LaunchBaseZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LavaReefZone.Act',('2')),
            ('SonicAndKnuckles3-Genesis','MarbleGardenZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','MushroomHillZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','SandopolisZone.Act',('1','2'))]
    '''
    info = None
    print('connecting to remote environment')
    #env = grc.RemoteEnv('tmp/sock')
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

    model = bebna()
    model.setup(env)
    obs = env.reset()
    done = False
    #while not done:
    reward = 0.0
    while True:
        obs = np.divide(obs, 255.0)
        #print(obs)
        #input()
        np.roll(obs, 1, axis=-1)
        action, genome_processed = model.process_data(obs, reward=reward, info=info)
        #obs, rew, done, info = env.step(env.action_space.sample())
        obs, reward, done, info = env.step(action)

        #print('reward: {}'.format(reward))
        #print(info)
        #print('Info: {}'.format(info))
        #model.animate_data(obs, obs)
        env.render()
        #if done:
        #    model.reset()
        #if model.population_done:
        #    model.population_done = False
        #    model.reset_population()
        #    obs = env.reset()
        #    model.NEAT.extra_frames_earned = 0
        if done and model.processing_old_paths:
            #input("Done...")
            model.map_tracker.unstick(model.current_path, model.NEAT)
            model.reset_population()
            model.extra_frames_earned = 0
        if done or genome_processed:
            obs = env.reset()
            model.observations_added = False
            ##model.current_observation = 0
            model.NEAT.extra_frames_earned = 0
            info = None
            #model.reset()


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
