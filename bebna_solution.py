
import numpy as np
import tensorflow as tf

from tensorflow.keras.engine.topology import Layer

from retro_contest.local import make


import gym_remote.exceptions as gre

#from hashlib import sha1
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import json
import math



class bebna:
    def __init__(self, initial_random_actions=256, num_actions=12,
            num_frames_before_show=100):
        self.initial_random_actions = initial_random_actions
        self.num_frames_before_show = num_frames_before_show
        self.current_frame = 0
        self.num_actions = num_actions


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
        self.down_train = 5
        self.down_b_train = 6
        self.left_b_train = 7
        self.right_b_train = 8
        self.b_train = 9
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



    def setup(self, env, observation_hashes={}):
        self.env = env

        self.last_action = self.env.action_space.sample()
        self.last_hash = None
        self.current_random_actions = 0

        self.observations = np.zeros((16, 224, 320, 3))
        self.num_observations = 0
        self.unique_observations = np.zeros((16,224,320, 3))
        ## How different the unique observations are to each other.
        ## We want high numbers here, rather than low ones.
        self.similarity = np.zeros((16, 16))
        self.uniqueness = np.zeros(16)
        self.uniqueness_check = np.zeros(16)
        self.observation_hashes = observation_hashes
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
        self.fig, self.ax = plt.subplots(1,1)
        self.image = self.ax.imshow(self.encoder_images[:, :, :], animated=True)
        #self.fig.canvas.draw()


    def animate_data(self, screen_frame, predicted_frame):
        #print(screen_frame[0][0][0], screen_frame[0][0][1], screen_frame[0][0][2])
        #print(np.amax(screen_frame))
        #screen_frame = screen_frame[:,:,::-1]
        predicted_frame = predicted_frame[:,:,::-1]
        ## Set top half of the encoder images to be the input frame.
        self.encoder_images[:self.img_height,:,:] = screen_frame.astype(int)
        self.encoder_images[self.img_height:, :,:] = screen_frame#predicted_frame
        self.image.set_data(self.encoder_images[:,:,:])
        #self.fig.canvas.draw()

    def save_image(self, data, hash, info, f):
        img_to_save = Image.fromarray(data, mode='RGB')
        img_to_save.save('frames/' + str(hash) + '.bmp')
        f[str(hash)] = info


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
            new_data[self.down_action] = True
        if data == self.down_b_train:
            new_data[self.down_action] = True
            new_data[self.b_action] = True
        if data == self.b_train:
            new_data[self.b_action] = True
        return new_data

    def process_data(self, obs, info={}, f=None):
        ## This was the old hash, we are now using distance hashes.
        #temp_hash = sha1(obs).hexdigest()
        ## New hashing method.
        temp_hash = imagehash.average_hash(Image.fromarray(obs, mode='RGB'))
        ## We want to do random things for a bit, but not too long. Then move
        ## on to actually predicting data.
        if self.current_random_actions < self.initial_random_actions:
            output_data = self.env.action_space.sample()
            self.current_random_actions += 1
        else:
            ## TODO: Actually do something :P
            #output_data = self.actuator_model.predict(np.expand_dims(obs, axis=0))[0]
            ##print('output_data final: {}'.format(output_data))
            #output_data = np.argmax(output_data)
            #output_data = self.convert_to_action_space(output_data)

            if np.array_equal(self.last_action, np.array([1, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0])):
                output_data = np.array([0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0])
            else:
                output_data = np.array([1, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0])


        ## Add this observation to the set of ones seen.
        ## Update the network, save the output from this, and return the new set
        ## of instructions.
        self.add_observation(obs, temp_hash, info, f)
        if temp_hash in self.observation_hashes:
            self.observation_hashes[temp_hash] += 1
        else:
            self.observation_hashes[temp_hash] = 1
        ##print("output_data: {}".format(output_data))
        self.last_action = output_data
        self.last_hash = temp_hash
        return output_data


    def add_observation(self, obs, temp_hash, info, f):
        ## Update the actuator model given the previous output and the current
        ## screen.
        self.num_observations += 1
        np.roll(self.observations, 1)
        np.copyto(self.observations[0], obs)
        found_new, temp_index = self.new_observation_check(temp_hash, obs, info, f)
        if found_new:
            np.copyto(self.unique_observations[temp_index], obs)
            #self.decoder_model.fit(self.unique_observations,
            #        self.unique_observations, verbose=0)
        elif self.num_observations < 16 :
            ## Train the autoencoder with our unique observations.
            np.roll(self.unique_observations, 1)
            np.copyto(self.unique_observations[0], obs)
            #self.decoder_model.fit(self.unique_observations,
            #        self.unique_observations, epochs=2, verbose=0)
        #self.decoder_model.fit(self.unique_observations,
        #        self.unique_observations, epochs=2, verbose=0)
        #self.animate_data(obs, self.decoder_eval_model.predict(np.expand_dims(obs, axis=0))[0].astype(int))


        temp_hash = self.last_hash
        temp_action = np.array(self.convert_to_train_space(self.last_action))
        if temp_hash in self.observation_hashes:
            ##temp_action = 1 - temp_action
            temp_index = np.argmax(temp_action)
            new_action = np.zeros(10)
            for i in self.opposites[temp_index]:
                new_action[i] = 1
            temp_action = new_action
        ##print("last_action: {}".format(self.last_action))
        ##print("last_train: {}".format(temp_action))
        ##print("last_train shape: {}".format(temp_action.shape))
        self.actuator_model.fit(np.expand_dims(self.observations[1], axis=0), np.expand_dims(temp_action, axis=0),verbose=0)


    def mse(self, index):
        ## Use the frame in slot zero of self.observations and the index item
        ## in self.unique_observations.
        return ((self.observations[0] -
               self.unique_observations[index]) ** 2).mean(axis=None)

    def mse_first(self, index_1, index_2):
        return ((self.unique_observations[index_1] -
               self.unique_observations[index_2]) ** 2).mean(axis=None)


    def new_observation_check(self, current_hash, obs, info, f):
        ## Only check if we have seen 16 observations.
        if self.num_observations > 16:
            ## We need to check how different our observation is.
            ## Keep track of the variation between all of them as we go.
            ## If we run across at least one that has a lower total variation
            ## then return the index of the lowest total variation and True.
            ## TODO: Do this.
            ## Check if it is a new observation. We don't need to check for old
            ## ones.
            if current_hash not in self.observation_hashes:
                self.save_image(obs, current_hash, info, f)
                for i in range(16):
                    self.uniqueness_check[i] = self.mse(i)
                temp_uniqueness = np.sum(self.uniqueness_check)
                current_min_uniqueness_index = np.argmin(self.uniqueness)
                current_min_uniqueness = self.uniqueness[
                        current_min_uniqueness_index]
                if temp_uniqueness > current_min_uniqueness:
                    ## We found one that is less unique than our current frame.
                    ## Replace the uniquness value in the system minux the
                    ## mse between the new and old frame.
                    self.uniqueness[
                            current_min_uniqueness_index] = temp_uniqueness - self.uniqueness_check[current_min_uniqueness_index]
                    for i in range(16):
                        ## Replace the difference values for each other frame.
                        if i != current_min_uniqueness_index:
                            self.similarity[i][current_min_uniqueness_index] = self.uniqueness_check[i]
                    return(True, current_min_uniqueness_index)
            return (False, 0)
        else:
            ## We should just put the new item in the unique_observations array.
            if self.num_observations == 16:
                ## We have now filled up the unique matrix. Calculate
                ## the differences between each unique observationself.
                for temp_key, temp_observation in enumerate(
                        self.unique_observations):
                    for i in range(temp_key, self.unique_observations.shape[0]):
                        temp_similarity = self.mse_first(temp_key, i)
                        self.similarity[temp_key][i] = temp_similarity
                        self.similarity[i][temp_key] = temp_similarity
                for i in range(16):
                    self.uniqueness[i] = np.sum(self.similarity[i])
            return (False, 0)

    def check_for_uniqueness(self, obs):
        for item in self.observations[-10:]:
            if np.array_equal(obs, item):
                return True
        return False
    '''
    def uniqueness_loss(self, X_true, X_pred):
        last_hash = self.last_hash
        temp_loss = 0
        if last_hash in self.observation_hashes:
            temp_loss += 10.0 * self.observation_hashes[last_hash]
        return tf.constant(temp_loss, dtype=float32)
    '''

    def make_encoder_decoder(self):
        with self.session as sess:
            try:
                #self.encoder_model = tf.keras.models.load_model('models/encoder_model')
                self.encoder_frozen_model = tf.keras.models.load_model('models/encoder_model.h5')
                for layer in self.encoder_frozen_model.layers:
                    layer.trainable=False

                #self.decoder_model = tf.keras.models.load_model('models/decoder_model')
                self.decoder_frozen_model = tf.keras.models.load_model('models/decoder_model.h5')
                for layer in self.decoder_frozen_model.layers:
                    layer.trainable=False

                #self.puzzle_solver_model = tf.keras.models.load_model('models/puzzle_model.h5')
                #self.decoder_eval_model = tf.keras.models.load_model('models/decoder_eval_model.h5')

            except:
                self.encoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                        )
                self.encoder_convolution_0 = tf.keras.layers.Conv2D(16,kernel_size=(4, 4),strides=(4,4), padding='valid')(self.encoder_input)
                self.encoder_convolution_1 = tf.keras.layers.Conv2D(32,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.encoder_convolution_0)
                self.encoder_convolution_2 = tf.keras.layers.Conv2D(64,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.encoder_convolution_1)
                self.encoder_convolution_3 = tf.keras.layers.Conv2D(128,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.encoder_convolution_2)
                self.encoder_flat = tf.keras.layers.Flatten()(
                        self.encoder_convolution_3)
                self.encoder_encoding_layer = tf.keras.layers.Dense(256,
                        )(self.encoder_flat)
                self.encoder_model = tf.keras.models.Model(
                        self.encoder_input, self.encoder_encoding_layer)

                ## Decoder
                ## The encoder segment should be trainable while we are training the
                ## autoencoder so we do not deactivate them here.
                ## Secondary path is based on: https://arxiv.org/pdf/1603.09246.pdf
                ## Secondary path for the decoder. Plan of action:
                ## We are going to slice the image up into units of h:112 w:160
                ## This makes a 4 by 4 grid of the image.
                ## Make a list from zero to 3.
                ## Randomly shuffle that list.

                self.decoder_sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.1,
                        decay=0.1)

                self.puzzle_solver_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                        name='puzzle_encoder_input')
                self.decoder_input = tf.keras.layers.Input(shape=(224, 320, 3,),
                        name='decoder_encoder_input')
                self.puzzle_encoder_layer = self.encoder_model(self.puzzle_solver_input)
                self.decoder_encoder_layer = self.encoder_model(self.decoder_input)

                self.puzzle_solver_layer_1 = tf.keras.layers.Dense(64)(self.puzzle_encoder_layer)
                self.puzzle_solver_layer_2 = tf.keras.layers.Dense(32)(self.puzzle_solver_layer_1)
                self.puzzle_solver_layer_3 = tf.keras.layers.Dense(24)(self.puzzle_solver_layer_2)
                self.puzzle_solver_output = tf.keras.layers.Dense(24, name='puzzle_output')(self.puzzle_solver_layer_3)
                self.puzzle_solver_model = tf.keras.models.Model(self.puzzle_solver_input, self.puzzle_solver_output)
                self.puzzle_solver_model.compile(optimizer='adam',loss={'puzzle_output':'categorical_crossentropy'})


                ## This next line is used if we ever want to be able to run the
                ## decoder by itself. Probably not something we need but it is
                ## there anyways.
                ##self.decoder_input = tf.layers.Input(128, name="decoder_input")
                self.decoder_decoding_layer = tf.keras.layers.Dense(
                        8960)(self.decoder_encoder_layer)
                self.decoder_reshaped = tf.keras.layers.Reshape(
                        target_shape=(7, 10, 128))(self.decoder_decoding_layer)
                self.decoder_deconvolution_1 = tf.keras.layers.Conv2DTranspose(256,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_reshaped)
                self.decoder_deconvolution_1_2 = tf.keras.layers.Conv2DTranspose(256,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_reshaped)
                self.decoder_deconvolution_2 = tf.keras.layers.Conv2DTranspose(128,
                        kernel_size=(2,2), strides=(2,2), padding='valid',
                        activation='relu')(self.decoder_deconvolution_1)
                self.decoder_deconvolution_2_2 = tf.keras.layers.Conv2DTranspose(128,
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

                self.decoder_model = tf.keras.models.Model([self.decoder_input],
                        [self.decoder_output])
                self.decoder_model.compile(optimizer='adam', loss='mse')
                print(self.decoder_model.summary())

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
                self.decoder_decoding_layer.trainable = False
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

                print(self.encoder_model.summary())

                #self.encoder_sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0,
                #        decay=0.0, nesterov=True)
                self.encoder_model.compile(optimizer='adam', loss='mse')


            ## Actuator code.
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

            tf.initialize_all_variables().run()

## MDN Stuff

def get_mixture_coef(output, numComonents=24, outputDim=1):
    out_pi = output[:,:numComonents]
    out_sigma = output[:,numComonents:2*numComonents]
    out_mu = output[:,2*numComonents:]
    out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])
    return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)

def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
return loss

class MDN(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, inputShape):
        return (inputShape[0], self.outputDim)





def main():

    envs = [('SonicTheHedgehog-Genesis','GreenHillZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','LabyrinthZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','MarbleZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','ScrapBrainZone.Act',('1','2')),
            ('SonicTheHedgehog-Genesis','SpringYardZone.Act',('1','2','3')),
            ('SonicTheHedgehog-Genesis','StarLightZone.Act',('1','2','3')),
            ('SonicTheHedgehog2-Genesis','AquaticRuinZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','CasinoNightZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','ChemicalPlantZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','EmeraldHillZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','HillTopZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','MetropolisZone.Act',('1','2','3')),
            ('SonicTheHedgehog2-Genesis','MysticCaveZone.Act',('1','2')),
            ('SonicTheHedgehog2-Genesis','OilOceanZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','AngelIslandZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','CarnivalNightZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','DeathEggZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','FlyingBatteryZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','HydrocityZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','IcecapZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LaunchBaseZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','LavaReefZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','MarbleGardenZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','MushroomHillZone.Act',('1','2')),
            ('SonicAndKnuckles3-Genesis','SandopolisZone.Act',('1','2'))]

    for j in envs:
        for k in range(len(j[2])):
            env = make(game='SonicAndKnuckles3-Genesis', state='SandopolisZone.Act2')

            model = bebna()
            model.setup(env)
            obs = env.reset()
            done == False
            while not done:
                action = model.process_data(obs, info)
                #obs, rew, done, info = env.step(env.action_space.sample())
                obs, rew, done, info = env.step(action)
                #print('Info: {}'.format(info))
                #model.animate_data(obs, obs)
                env.render()
                if done:
                    obs = env.reset()


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
